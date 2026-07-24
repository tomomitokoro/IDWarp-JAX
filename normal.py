"""Surface-normal and nodal-area calculations for JAX mesh deformation.

This module computes unit normals and area weights at surface mesh nodes from
polygonal face connectivity. All numerical operations use JAX and remain
compatible with JAX transformations and automatic differentiation.

The surface topology is represented by:

``pts``         Surface-node coordinates with shape ``(n_surface, 3)``.
``conn``        Flattened face connectivity. The node indices for all faces are stored consecutively in a one-dimensional array.
``faceSizes``   Number of nodes in each face. Together with ``conn``, this defines the start and end positions of every face.
"""

import jax
import jax.numpy as jnp


def create_face_pointer(faceSizes):
    """Construct offsets into flattened face connectivity.

    Parameters
    ----------
    faceSizes: Number of nodes in each surface face, with shape ``(n_faces,)``.

    Returns
    -------
    facePtr
        Connectivity offsets with shape ``(n_faces + 1,)``. The connectivity entries for face ``i`` are stored in::
        conn[facePtr[i] : facePtr[i + 1]]

    Examples
    --------
    For ``faceSizes = [3, 4, 5]``, the returned pointer is ``[0, 3, 7, 12]``.
    """
    faceSizes = jnp.asarray(faceSizes, dtype=jnp.int32)

    facePtr = jnp.concatenate(
        [
            jnp.array([0], dtype=jnp.int32),
            jnp.cumsum(faceSizes),
        ]
    )

    return facePtr

@jax.jit
def compute_node_normals(
    pts,
    conn,
    faceSizes,
    eps=1e-30,
):
    """Compute unit node normals and nodal area weights.
    Face normals are computed from the ordered polygon vertices using
    ``N_f = sum_k (x_k cross x_{k+1})``.

        The corresponding face area is
    ``A_f = 0.5 * ||N_f||``.

    Each face contributes an equal fraction of its area to each of its corner
    nodes. Node normals are formed by averaging the adjacent unit face normals
    using these area contributions as weights, then normalizing the result.

    Parameters
    ----------
    pts:        Surface-node coordinates with shape ``(n_surface, 3)``.
    conn:       Flattened local face connectivity with shape ``(n_connectivity,)``.
    faceSizes:  Number of nodes in each face, with shape ``(n_faces,)``.
    eps:        Small positive regularization used in norms and divisions.

    Returns
    -------
    normals: Unit normal vector at each surface node, with shape ``(n_surface, 3)``.
    Ai:      Accumulated area weight at each surface node, with shape ``(n_surface,)``.
    """
    pts = jnp.asarray(pts)
    conn = jnp.asarray(conn, dtype=jnp.int32)
    faceSizes = jnp.asarray(faceSizes, dtype=jnp.int32)
    

    nSurf = pts.shape[0]                
    nFace = faceSizes.shape[0]          
    nConn = conn.shape[0]               

    # Build offsets into the flattened connectivity array.
    facePtr = create_face_pointer(faceSizes) 

    cornerIds = jnp.arange(nConn, dtype=jnp.int32)
    faceIds = jnp.searchsorted(
        facePtr[1:],
        cornerIds,
        side="right",
    )
    
    # For every face corner, find the following corner in the same face.
    # The final corner wraps back to the first corner of that face.
    faceStart = facePtr[faceIds]
    faceEnd = facePtr[faceIds + 1]
    
    nextCornerIds = jnp.where(
        cornerIds + 1 < faceEnd,
        cornerIds + 1,
        faceStart,
    )

    # Current and next node indices for each directed polygon edge.
    nodeIds = conn
    nextNodeIds = conn[nextCornerIds]
    
    # Coordinates at the ends of each directed polygon edge.
    x0 = pts[nodeIds]
    x1 = pts[nextNodeIds]

    # Each edge contributes x_k cross x_{k+1} to its face normal.
    edgeCross = jnp.cross(x0, x1)

    # Sum edge contributions separately for every face.
    rawFaceNormal = jax.ops.segment_sum(
        edgeCross,
        faceIds,
        num_segments=nFace,
    )

    # Compute face areas and unit face normals.
    rawNorm = jnp.sqrt(
        jnp.sum(rawFaceNormal * rawFaceNormal, axis=1) + eps
    )

    faceArea = 0.5 * rawNorm
    faceNormal = rawFaceNormal / rawNorm[:, None]

    # Divide each valid face area equally among its corner nodes.
    c_f = faceSizes.astype(pts.dtype)

    validFace = faceSizes >= 3

    dA = jnp.where(
        validFace,
        faceArea / c_f,
        0.0,
    )

    # Gather the area and unit-normal contribution for every face corner.
    cornerDA = dA[faceIds]
    cornerFaceNormal = faceNormal[faceIds]

    cornerNormalContribution = cornerDA[:, None] * cornerFaceNormal

    # Accumulate all adjacent-face contributions at each surface node.
    normalSum = jnp.zeros((nSurf, 3), dtype=pts.dtype)
    areaSum = jnp.zeros((nSurf,), dtype=pts.dtype)

    normalSum = normalSum.at[nodeIds].add(cornerNormalContribution)
    areaSum = areaSum.at[nodeIds].add(cornerDA)

    # Form the area-weighted average normal at each node.
    normals = normalSum / (areaSum[:, None] + eps)

    # Normalize the averaged node normals to unit length.
    normalNorm = jnp.sqrt(
        jnp.sum(normals * normals, axis=1, keepdims=True) + eps
    )

    normals = normals / normalNorm

    Ai = areaSum

    return normals, Ai


@jax.jit
def get_normals_Ai(
    pts0,
    pts,
    conn,
    faceSizes,
    eps=1e-30,
):
    """Compute original and deformed node normals and original area weights.

    Parameters
    ----------
    pts0        Original surface-node coordinates, equivalent to ``Xs0``, with shape ``(n_surface, 3)``.
    pts         Deformed surface-node coordinates, equivalent to ``Xs``, with shape ``(n_surface, 3)``.
    conn        Flattened local face connectivity.
    faceSizes   Number of nodes in each face.
    eps         Small positive regularization used by the normal calculations.

    Returns
    -------
    normals0    Unit node normals of the original surface.
    normals     Unit node normals of the deformed surface.
    Ai          Node area weights computed from the original surface.
    """    

    normals0, Ai = compute_node_normals(
        pts0,
        conn,
        faceSizes,
        eps,
    )

    normals, _ = compute_node_normals(
        pts,
        conn,
        faceSizes,
        eps,
    )

    return normals0, normals, Ai


