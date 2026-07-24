"""Public array-based interface for JAX mesh deformation.

This module contains no OpenFOAM, file-system, command-line, or NumPy I/O
logic. The caller prepares the mesh arrays and supplies a displacement of the
original surface nodes. :func:`deform_mesh` then performs the complete JAX
mesh-deformation calculation:

1. construct the target surface coordinates,
2. constrain surface nodes originally on a symmetry plane when requested,
3. compute original and deformed surface normals,
4. compute local rotation matrices and translation vectors,
5. call the unified volume-warp interface, and
6. overwrite prescribed surface-boundary coordinates exactly.

The returned value remains a JAX array. Synchronization, conversion to NumPy,
and any mesh-file output are responsibilities of the calling application.

For JVP or VJP use, :func:`make_deformation_function` fixes the mesh and
configuration arguments and returns the one-input mapping expected by the
generic utilities in ``derivative.py``::

    surface_displacement -> deformed volume coordinates
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp

import normal
import rotation
import symmetry
import warp
from symmetry import (
    SYMMETRY_APPROXIMATE,
    SYMMETRY_NONE,
)

Array = jax.Array

__all__ = ["deform_mesh", "make_deformation_function"]


def deform_mesh(
    Xv0,
    Xs0,
    surface_displacement,
    conn,
    face_sizes,
    surface_global_ids,
    *,
    Ldef: float = 1.0,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    normal_eps: float = 1.0e-30,
    rotation_eps: float = 1.0e-12,
    warp_eps: float = 1.0e-30,
    volume_chunk_size: int = 512,
    surface_block_size: int = 1024,
    symmetry_mode: str | None = SYMMETRY_NONE,
    symmetry_plane_point=None,
    symmetry_plane_normal=None,
    symmetry_tolerance: float = 1.0e-10,
    symmetry_length_scale: float | None = None,
) -> Array:
    """Deform a volume mesh from prescribed surface-node displacements.

    Parameters
    ----------
    Xv0:                    Original volume coordinates with shape ``(n_volume, 3)``.
    Xs0:                    Original surface coordinates with shape ``(n_surface, 3)``.
    surface_displacement:   Displacement measured from ``Xs0``, with shape ``(n_surface, 3)``.
    conn:                   Flattened local surface-face connectivity.
    face_sizes:             Number of nodes in each surface face.
    surface_global_ids:     Mapping from each local surface-node index to its corresponding volume-mesh point index.
    Ldef:                   Reference length used by the IDW weights.
    aExp, bExp, alpha:      IDW weighting parameters.
    normal_eps:             Numerical regularization used by surface-normal calculations.
    rotation_eps:           Numerical tolerance used by local rotation calculations.
    warp_eps:               Numerical regularization used by the warp and symmetry geometry.
    volume_chunk_size:      Number of volume points handled by each Python-level warp chunk.
    surface_block_size:     Number of surface points handled by each compiled warp block.
    symmetry_mode:          One of ``"nonesym"``, ``"exactsym"``, or ``"approxsym"``.
    symmetry_plane_point:   Definition of the symmetry plane. Both point and normal are required by exact and approximate symmetry modes.
    symmetry_plane_normal:  Definition of the symmetry plane. Both point and normal are required by exact and approximate symmetry modes.
    symmetry_tolerance:     Distance tolerance used to identify surface points that were originally on the symmetry plane.
    symmetry_length_scale:  Recovery distance for approximate symmetry. Required when ``symmetry_mode="approxsym"``.

    Returns
    -------
    Array
        Deformed volume coordinates with shape ``(n_volume, 3)``.

    Notes
    -----
    The caller is responsible for supplying corresponding arrays, consistent
    floating-point dtypes, and valid fixed-topology connectivity. The function
    intentionally does not call ``block_until_ready`` or ``jax.device_get`` so
    that it remains composable with JAX transformations such as JVP and VJP.
    """
    mode = symmetry.normalize_symmetry_mode(symmetry_mode)

    if volume_chunk_size <= 0:
        raise ValueError("volume_chunk_size must be positive")
    if surface_block_size <= 0:
        raise ValueError("surface_block_size must be positive")

    if mode != SYMMETRY_NONE:
        if symmetry_plane_point is None or symmetry_plane_normal is None:
            raise ValueError(
                "symmetry_plane_point and symmetry_plane_normal are required "
                "for exact and approximate symmetry modes"
            )

    if mode == SYMMETRY_APPROXIMATE:
        if symmetry_length_scale is None:
            raise ValueError(
                "symmetry_length_scale is required for 'approxsym'"
            )
        if symmetry_length_scale <= 0.0:
            raise ValueError(
                "symmetry_length_scale must be positive for 'approxsym'"
            )

    # Use Xv0 as the floating-point dtype reference. Integer topology arrays
    # are converted explicitly because they are used as JAX indices.
    Xv0 = jnp.asarray(Xv0)
    Xs0 = jnp.asarray(Xs0, dtype=Xv0.dtype)
    surface_displacement = jnp.asarray(
        surface_displacement,
        dtype=Xv0.dtype,
    )
    conn = jnp.asarray(conn, dtype=jnp.int32)
    face_sizes = jnp.asarray(face_sizes, dtype=jnp.int32)
    surface_global_ids = jnp.asarray(
        surface_global_ids,
        dtype=jnp.int32,
    )

    # Construct the requested deformed surface.
    Xs_unconstrained = Xs0 + surface_displacement

    # Surface nodes that belonged to a symmetry plane in the reference mesh
    # remain on that plane. Keeping this operation inside the public mapping
    # ensures that JVP/VJP includes the surface constraint.
    if mode == SYMMETRY_NONE:
        Xs = Xs_unconstrained
        plane_point = None
        plane_normal = None
    else:
        plane_point = jnp.asarray(
            symmetry_plane_point,
            dtype=Xv0.dtype,
        )
        plane_normal = jnp.asarray(
            symmetry_plane_normal,
            dtype=Xv0.dtype,
        )

        Xs = symmetry.constrain_points_originally_on_plane(
            original_points=Xs0,
            target_points=Xs_unconstrained,
            plane_point=plane_point,
            plane_normal=plane_normal,
            tolerance=symmetry_tolerance,
            eps=warp_eps,
        )

    normals0, Ai = normal.compute_node_normals(
        Xs0,
        conn,
        face_sizes,
        eps=normal_eps,
    )

    normals, _ = normal.compute_node_normals(
        Xs,
        conn,
        face_sizes,
        eps=normal_eps,
    )

    M, b = rotation.get_MB_rotation(
        Xs0,
        Xs,
        normals0,
        normals,
        eps=rotation_eps,
    )

    Xv = warp.deformed_volume_points(
        Xv0=Xv0,
        Xs0=Xs0,
        Ai=Ai,
        M=M,
        b=b,
        Ldef=Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        eps=warp_eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_mode=mode,
        symmetry_plane_point=plane_point,
        symmetry_plane_normal=plane_normal,
        symmetry_length_scale=symmetry_length_scale,
    )

    # The prescribed surface is an exact boundary condition rather than an
    # IDW approximation. This indexed update is differentiable in JAX.
    return Xv.at[surface_global_ids].set(Xs)


def make_deformation_function(
    Xv0,
    Xs0,
    conn,
    face_sizes,
    surface_global_ids,
    *,
    Ldef: float = 1.0,
    aExp: float = 3.0,
    bExp: float = 5.0,
    alpha: float = 0.25,
    normal_eps: float = 1.0e-30,
    rotation_eps: float = 1.0e-12,
    warp_eps: float = 1.0e-30,
    volume_chunk_size: int = 512,
    surface_block_size: int = 1024,
    symmetry_mode: str | None = SYMMETRY_NONE,
    symmetry_plane_point=None,
    symmetry_plane_normal=None,
    symmetry_tolerance: float = 1.0e-10,
    symmetry_length_scale: float | None = None,
) -> Callable[[Array], Array]:
    """Create a one-input mesh-deformation function for JAX transforms.

    The returned function has the signature::

        deformation_function(surface_displacement) -> Xv

    Mesh arrays and configuration values are captured without performing a
    deformation. The actual calculation occurs only when the returned function
    is called, including when it is passed to ``jax.jvp`` or ``jax.vjp`` through
    the generic utilities in ``derivative.py``.
    """

    def deformation_function(surface_displacement: Array) -> Array:
        return deform_mesh(
            Xv0=Xv0,
            Xs0=Xs0,
            surface_displacement=surface_displacement,
            conn=conn,
            face_sizes=face_sizes,
            surface_global_ids=surface_global_ids,
            Ldef=Ldef,
            aExp=aExp,
            bExp=bExp,
            alpha=alpha,
            normal_eps=normal_eps,
            rotation_eps=rotation_eps,
            warp_eps=warp_eps,
            volume_chunk_size=volume_chunk_size,
            surface_block_size=surface_block_size,
            symmetry_mode=symmetry_mode,
            symmetry_plane_point=symmetry_plane_point,
            symmetry_plane_normal=symmetry_plane_normal,
            symmetry_tolerance=symmetry_tolerance,
            symmetry_length_scale=symmetry_length_scale,
        )

    return deformation_function
