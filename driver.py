"""Public array-based interface for JAX mesh deformation.

The caller supplies corresponding mesh arrays and a displacement of the
original surface nodes. 
OpenFOAM I/O and command-line handling remain outside this module.

Two public entry points are provided:

``deform_mesh``
    Self-contained evaluation for a single deformation.
``make_deformation_function``
    Prepare fixed mesh data once and return the one-input mapping required by
    JAX JVP and VJP operations.
"""

from __future__ import annotations

from collections.abc import Callable

import jax

from deformation_core import deform_prepared_mesh
from deformation_preparation import (
    DeformationOptions,
    prepare_deformation_data,
    validate_deformation_configuration,
)
from symmetry import SYMMETRY_NONE

Array = jax.Array

__all__ = ["deform_mesh", "make_deformation_function"]


def _make_options(
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
    symmetry_length_scale: float | None,
) -> DeformationOptions:
    """Collect numerical settings passed unchanged to the deformation core."""
    return DeformationOptions(
        Ldef=Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        normal_eps=normal_eps,
        rotation_eps=rotation_eps,
        warp_eps=warp_eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_length_scale=symmetry_length_scale,
    )


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
    """Deform one volume mesh from prescribed surface-node displacements."""
    mode = validate_deformation_configuration(
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_mode=symmetry_mode,
        symmetry_plane_point=symmetry_plane_point,
        symmetry_plane_normal=symmetry_plane_normal,
        symmetry_length_scale=symmetry_length_scale,
    )

    prepared = prepare_deformation_data(
        Xv0=Xv0,
        Xs0=Xs0,
        conn=conn,
        face_sizes=face_sizes,
        surface_global_ids=surface_global_ids,
        normal_eps=normal_eps,
        warp_eps=warp_eps,
        symmetry_mode=mode,
        symmetry_plane_point=symmetry_plane_point,
        symmetry_plane_normal=symmetry_plane_normal,
        symmetry_tolerance=symmetry_tolerance,
    )

    options = _make_options(
        Ldef=Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        normal_eps=normal_eps,
        rotation_eps=rotation_eps,
        warp_eps=warp_eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_length_scale=symmetry_length_scale,
    )
    return deform_prepared_mesh(prepared, surface_displacement, options)


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
    """Return ``surface_displacement -> Xv`` with fixed data prepared once."""
    mode = validate_deformation_configuration(
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_mode=symmetry_mode,
        symmetry_plane_point=symmetry_plane_point,
        symmetry_plane_normal=symmetry_plane_normal,
        symmetry_length_scale=symmetry_length_scale,
    )

    prepared = prepare_deformation_data(
        Xv0=Xv0,
        Xs0=Xs0,
        conn=conn,
        face_sizes=face_sizes,
        surface_global_ids=surface_global_ids,
        normal_eps=normal_eps,
        warp_eps=warp_eps,
        symmetry_mode=mode,
        symmetry_plane_point=symmetry_plane_point,
        symmetry_plane_normal=symmetry_plane_normal,
        symmetry_tolerance=symmetry_tolerance,
    )

    options = _make_options(
        Ldef=Ldef,
        aExp=aExp,
        bExp=bExp,
        alpha=alpha,
        normal_eps=normal_eps,
        rotation_eps=rotation_eps,
        warp_eps=warp_eps,
        volume_chunk_size=volume_chunk_size,
        surface_block_size=surface_block_size,
        symmetry_length_scale=symmetry_length_scale,
    )

    def deformation_function(surface_displacement: Array) -> Array:
        return deform_prepared_mesh(
            prepared,
            surface_displacement,
            options,
        )

    return deformation_function
