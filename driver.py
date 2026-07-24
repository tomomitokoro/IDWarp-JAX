"""Public array-based interface for JAX mesh deformation.

This module contains no OpenFOAM, file-system, or command-line logic. The
caller prepares the mesh arrays and supplies a displacement of the original
surface nodes. :func:`deform_mesh` then performs the complete JAX
mesh-deformation calculation:

1. construct the target surface coordinates,
2. constrain surface nodes originally on a symmetry plane when requested,
3. compute original and deformed surface normals,
4. compute local rotation matrices and translation vectors,
5. warp only volume points that are not prescribed surface points, and
6. insert the warped non-surface points and prescribed surface coordinates into the
   full volume-coordinate array.

The returned value remains a JAX array. Synchronization, conversion to NumPy,
and any mesh-file output are responsibilities of the calling application.

For JVP or VJP use, :func:`make_deformation_function` fixes the mesh and
configuration arguments and returns the one-input mapping expected by the
generic utilities in ``derivative.py``::

    surface_displacement -> deformed volume coordinates

``make_deformation_function`` also prepares the fixed non-surface volume-point
index array, original surface normals, original area weights, reusable normal-topology
indices, and the mask of surface points originally on the symmetry plane
once. These reference
values are captured and reused by every deformation, JVP, and VJP evaluation.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

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


def _make_warp_global_ids(
    n_volume: int,
    surface_global_ids,
) -> Array:
    """Return fixed global IDs for volume points outside the surface.

    NumPy is used only for this one-time topology operation. The resulting
    integer index array is converted back to a JAX array and can then be reused
    by GPU-resident deformation calculations.
    """
    surface_ids_np = np.asarray(
        jax.device_get(surface_global_ids),
        dtype=np.int64,
    ).reshape(-1)

    if np.any(surface_ids_np < 0) or np.any(surface_ids_np >= n_volume):
        raise ValueError(
            "surface_global_ids contains an index outside the volume mesh"
        )

    if np.unique(surface_ids_np).size != surface_ids_np.size:
        raise ValueError("surface_global_ids must not contain duplicates")

    surface_mask = np.zeros(n_volume, dtype=bool)
    surface_mask[surface_ids_np] = True

    warp_global_ids_np = np.flatnonzero(~surface_mask).astype(
        np.int32,
        copy=False,
    )

    return jnp.asarray(warp_global_ids_np, dtype=jnp.int32)


def _deform_mesh_with_warp_ids(
    Xv0: Array,
    Xs0: Array,
    surface_displacement: Array,
    conn: Array,
    face_sizes: Array,
    surface_global_ids: Array,
    warp_global_ids: Array,
    normals0: Array,
    Ai: Array,
    surface_on_plane: Array | None,
    normal_topology: normal.NormalTopology,
    *,
    Ldef: float,
    aExp: float,
    bExp: float,
    alpha: float,
    normal_eps: float,
    rotation_eps: float,
    warp_eps: float,
    volume_chunk_size: int,
    surface_block_size: int,
    symmetry_mode: str,
    symmetry_plane_point,
    symmetry_plane_normal,
    symmetry_length_scale: float | None,
) -> Array:
    """Deform a mesh using precomputed fixed reference data."""
    mode = symmetry.normalize_symmetry_mode(symmetry_mode)

    # Construct the requested deformed surface.
    Xs_unconstrained = Xs0 + surface_displacement

    # Surface nodes that belonged to a symmetry plane in the reference mesh
    # remain on that plane. Keeping this operation inside the differentiable
    # mapping ensures that JVP/VJP includes the surface constraint.
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

        projected_surface = symmetry.project_points_to_plane(
            Xs_unconstrained,
            plane_point,
            plane_normal,
            eps=warp_eps,
        )
        Xs = jnp.where(
            surface_on_plane[:, None],
            projected_surface,
            Xs_unconstrained,
        )

    # Original normals and area weights are fixed reference quantities.
    # They are supplied by the caller and reused here without recomputation.
    normals, _ = normal.compute_node_normals_from_topology(
        Xs,
        normal_topology,
        eps=normal_eps,
    )

    M, b = rotation.get_MB_rotation(
        Xs0,
        Xs,
        normals0,
        normals,
        eps=rotation_eps,
    )

    # Only points outside the prescribed surface enter the expensive IDW calculation. 
    # Surface points are inserted exactly afterward.
    Xv_non_surface = warp.deformed_volume_points(
        Xv0=Xv0[warp_global_ids],
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

    # Reconstruct the full volume array. The two index sets are disjoint:
    # warp_global_ids receives the IDW result and surface_global_ids receives
    # the prescribed surface coordinates exactly.
    Xv = Xv0.at[warp_global_ids].set(Xv_non_surface)
    return Xv.at[surface_global_ids].set(Xs)


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

    Prescribed surface points are excluded from the IDW calculation and assigned
    the target surface coordinates exactly.
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

    # A direct deform_mesh call has no persistent closure, so its fixed index
    # array is prepared for that call. Repeated applications should use
    # make_deformation_function, which computes and stores this array once.
    warp_global_ids = _make_warp_global_ids(
        Xv0.shape[0],
        surface_global_ids,
    )

    # A direct call is self-contained, so fixed reference geometry is
    # prepared once for this call. Repeated applications should use
    # make_deformation_function(), which retains these arrays for reuse.
    normal_topology = normal.prepare_normal_topology(
        conn,
        face_sizes,
    )
    normals0, Ai = normal.compute_node_normals_from_topology(
        Xs0,
        normal_topology,
        eps=normal_eps,
    )

    # A direct call is also self-contained for symmetry geometry. The mask is
    # fixed for this mesh and plane, but without a persistent closure it is
    # prepared once for this call.
    if mode == SYMMETRY_NONE:
        surface_on_plane = None
    else:
        surface_on_plane = symmetry.points_on_plane(
            Xs0,
            symmetry_plane_point,
            symmetry_plane_normal,
            symmetry_tolerance,
            eps=warp_eps,
        )

    return _deform_mesh_with_warp_ids(
        Xv0=Xv0,
        Xs0=Xs0,
        surface_displacement=surface_displacement,
        conn=conn,
        face_sizes=face_sizes,
        surface_global_ids=surface_global_ids,
        warp_global_ids=warp_global_ids,
        normals0=normals0,
        Ai=Ai,
        surface_on_plane=surface_on_plane,
        normal_topology=normal_topology,
        Ldef=Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        normal_eps=normal_eps,
        rotation_eps=rotation_eps,
        warp_eps=warp_eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_mode=mode,
        symmetry_plane_point=symmetry_plane_point,
        symmetry_plane_normal=symmetry_plane_normal,
        symmetry_length_scale=symmetry_length_scale,
    )


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

    The fixed non-surface volume-point IDs, normal-topology indices, original
    surface normals, original area weights, and symmetry-plane surface mask
    are prepared once and
    captured in the returned function. Subsequent deformation, JVP, and VJP
    evaluations reuse these arrays.
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

    # Convert and capture all fixed arrays before constructing the one-input
    # differentiable mapping.
    Xv0 = jnp.asarray(Xv0)
    Xs0 = jnp.asarray(Xs0, dtype=Xv0.dtype)
    conn = jnp.asarray(conn, dtype=jnp.int32)
    face_sizes = jnp.asarray(face_sizes, dtype=jnp.int32)
    surface_global_ids = jnp.asarray(
        surface_global_ids,
        dtype=jnp.int32,
    )

    if mode == SYMMETRY_NONE:
        plane_point = None
        plane_normal = None
        surface_on_plane = None
    else:
        plane_point = jnp.asarray(
            symmetry_plane_point,
            dtype=Xv0.dtype,
        )
        plane_normal = jnp.asarray(
            symmetry_plane_normal,
            dtype=Xv0.dtype,
        )

        # Fixed reference-geometry operation: identify once which original
        # Surface points belong to the symmetry plane.
        surface_on_plane = symmetry.points_on_plane(
            Xs0,
            plane_point,
            plane_normal,
            symmetry_tolerance,
            eps=warp_eps,
        )

    # Fixed topology operation: compute once on the CPU with NumPy, then retain
    # the JAX index array in the closure for all later GPU computations.
    warp_global_ids = _make_warp_global_ids(
        Xv0.shape[0],
        surface_global_ids,
    )

    # Reference surface geometry depends only on Xs0 and fixed topology.
    # Compute it once before constructing the differentiable one-input map.
    normal_topology = normal.prepare_normal_topology(
        conn,
        face_sizes,
    )
    normals0, Ai = normal.compute_node_normals_from_topology(
        Xs0,
        normal_topology,
        eps=normal_eps,
    )

    def deformation_function(surface_displacement: Array) -> Array:
        surface_displacement_jax = jnp.asarray(
            surface_displacement,
            dtype=Xv0.dtype,
        )

        return _deform_mesh_with_warp_ids(
            Xv0=Xv0,
            Xs0=Xs0,
            surface_displacement=surface_displacement_jax,
            conn=conn,
            face_sizes=face_sizes,
            surface_global_ids=surface_global_ids,
            warp_global_ids=warp_global_ids,
            normals0=normals0,
            Ai=Ai,
            surface_on_plane=surface_on_plane,
            normal_topology=normal_topology,
            Ldef=Ldef,
            aExp=aExp,
            bExp=bExp,
            alpha=alpha,
            normal_eps=normal_eps,
            rotation_eps=rotation_eps,
            warp_eps=warp_eps,
            volume_chunk_size=volume_chunk_size,
            surface_block_size=surface_block_size,
            symmetry_mode=mode,
            symmetry_plane_point=plane_point,
            symmetry_plane_normal=plane_normal,
            symmetry_length_scale=symmetry_length_scale,
        )

    return deformation_function