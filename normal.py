"""Surface-node normal calculations with reusable fixed topology.

The surface connectivity does not change during deformation. 
This module therefore separates one-time topology preparation 
from coordinate-dependent normal and area calculations.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

Array = jax.Array


class NormalTopology(NamedTuple):
    """Fixed indexing arrays used by the surface-normal calculation."""

    face_ptr: Array
    corner_ids: Array
    face_ids: Array
    face_start: Array
    face_end: Array
    next_corner_ids: Array
    node_ids: Array
    next_node_ids: Array
    face_sizes: Array


def create_face_pointer(face_sizes) -> Array:
    """Return the flattened-connectivity pointer for each face."""
    face_sizes = jnp.asarray(face_sizes, dtype=jnp.int32)
    return jnp.concatenate(
        (
            jnp.zeros((1,), dtype=jnp.int32),
            jnp.cumsum(face_sizes, dtype=jnp.int32),
        )
    )


# Backward-compatible alias used by older code.
create_face_pointer.__name__ = "create_face_pointer"


def prepare_normal_topology(conn, face_sizes) -> NormalTopology:
    """Prepare all coordinate-independent normal-calculation indices once.

    Parameters
    ----------
    conn
        Flattened local surface-face connectivity.
    face_sizes
        Number of nodes in each surface face.
    """
    conn = jnp.asarray(conn, dtype=jnp.int32)
    face_sizes = jnp.asarray(face_sizes, dtype=jnp.int32)

    n_conn = conn.shape[0]
    face_ptr = create_face_pointer(face_sizes)
    corner_ids = jnp.arange(n_conn, dtype=jnp.int32)
    face_ids = jnp.searchsorted(
        face_ptr[1:],
        corner_ids,
        side="right",
    )

    face_start = face_ptr[face_ids]
    face_end = face_ptr[face_ids + 1]
    next_corner_ids = jnp.where(
        corner_ids + 1 < face_end,
        corner_ids + 1,
        face_start,
    )

    node_ids = conn
    next_node_ids = conn[next_corner_ids]

    return NormalTopology(
        face_ptr=face_ptr,
        corner_ids=corner_ids,
        face_ids=face_ids,
        face_start=face_start,
        face_end=face_end,
        next_corner_ids=next_corner_ids,
        node_ids=node_ids,
        next_node_ids=next_node_ids,
        face_sizes=face_sizes,
    )


@jax.jit
def compute_node_normals_from_topology(
    pts,
    topology: NormalTopology,
    eps: float = 1.0e-30,
):
    """Compute nodal normals and area weights using prepared topology."""
    pts = jnp.asarray(pts)

    n_surface = pts.shape[0]
    n_face = topology.face_sizes.shape[0]

    x0 = pts[topology.node_ids]
    x1 = pts[topology.next_node_ids]

    edge_cross = jnp.cross(x0, x1)
    raw_face_normal = jax.ops.segment_sum(
        edge_cross,
        topology.face_ids,
        num_segments=n_face,
    )

    raw_norm = jnp.sqrt(
        jnp.sum(raw_face_normal * raw_face_normal, axis=1) + eps
    )
    face_area = 0.5 * raw_norm
    face_normal = raw_face_normal / raw_norm[:, None]

    corners_per_face = topology.face_sizes.astype(pts.dtype)
    valid_face = topology.face_sizes >= 3
    area_per_corner = jnp.where(
        valid_face,
        face_area / corners_per_face,
        0.0,
    )

    corner_area = area_per_corner[topology.face_ids]
    corner_face_normal = face_normal[topology.face_ids]
    corner_normal_contribution = (
        corner_area[:, None] * corner_face_normal
    )

    normal_sum = jnp.zeros((n_surface, 3), dtype=pts.dtype)
    area_sum = jnp.zeros((n_surface,), dtype=pts.dtype)
    normal_sum = normal_sum.at[topology.node_ids].add(
        corner_normal_contribution
    )
    area_sum = area_sum.at[topology.node_ids].add(corner_area)

    normals = normal_sum / (area_sum[:, None] + eps)
    normal_norm = jnp.sqrt(
        jnp.sum(normals * normals, axis=1, keepdims=True) + eps
    )
    normals = normals / normal_norm

    return normals, area_sum


def compute_node_normals(
    pts,
    conn,
    faceSizes,
    eps: float = 1.0e-30,
):
    """Backward-compatible self-contained normal calculation.

    Repeated deformation calls should instead prepare topology once with
    :func:`prepare_normal_topology` and call
    :func:`compute_node_normals_from_topology`.
    """
    topology = prepare_normal_topology(conn, faceSizes)
    return compute_node_normals_from_topology(pts, topology, eps=eps)


def get_normals_Ai(
    pts0,
    pts,
    conn,
    faceSizes,
    eps: float = 1.0e-30,
):
    """Compute original/deformed normals and original area weights."""
    topology = prepare_normal_topology(conn, faceSizes)
    normals0, Ai = compute_node_normals_from_topology(
        pts0,
        topology,
        eps=eps,
    )
    normals, _ = compute_node_normals_from_topology(
        pts,
        topology,
        eps=eps,
    )
    return normals0, normals, Ai