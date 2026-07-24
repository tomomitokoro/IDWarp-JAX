# IDWarp-JAX

A JAX-based implementation of inverse-distance-weighted volume-mesh deformation.

This repository provides an array-based mesh-deformation interface inspired by the mesh warping method used in IDWarp. The implementation is designed to preserve compatibility with JAX transformations and automatic differentiation.

The current implementation supports:

* mesh deformation without symmetry,
* exact symmetry-plane treatment,
* approximate symmetry-plane treatment,
* Jacobian-vector products,
* vector-Jacobian products, and
* basic derivative-consistency checks.

> [!IMPORTANT]
> The approximate symmetry mode (`approxsym`) and the derivative utilities are still under development. They have not yet been fully tested or validated for general mesh-deformation problems.

## Background and References

This implementation is based on the inverse-distance-weighted mesh-deformation approach described in the IDWarp literature.

* **IDWarp paper:** [Efficient Mesh Generation and Deformation for Aerodynamic Shape Optimization](https://doi.org/10.2514/1.J059491)
* **IDWarp GitHub repository:** [GitHub link](https://github.com/mdolab/idwarp.git)

This project is an independent JAX implementation and is not part of the original IDWarp repository.

## Repository Structure

```text
.
├── driver.py
├── deformation_core.py
├── deformation_preparation.py
├── derivative.py
├── normal.py
├── rotation.py
├── symmetry.py
├── warp.py
└── README.md
```

### `driver.py`

Provides the public array-based mesh-deformation interface.

The main functions are:

* `deform_mesh(...)`
* `make_deformation_function(...)`

### `deformation_preparation.py`

Prepares fixed mesh and reference-geometry data that can be reused across
multiple deformation evaluations.

Prepared data includes:
* non-surface volume-point indices,
* reusable surface-normal topology,
* original surface normals,
* original nodal area weights, and
* the symmetry-plane surface mask when symmetry is enabled.

### `deformation_core.py`

Contains the deformation-dependent calculation performed for each new surface
displacement.

It constructs the target surface, applies symmetry-plane constraints, computes
the deformed surface normals and local transformations, performs the IDW
volume deformation, and reconstructs the complete volume-coordinate array.

### `warp.py`

Contains the IDW volume-mesh deformation algorithms.

It supports three symmetry modes:

* `nonesym`
* `exactsym`
* `approxsym`

### `normal.py`

Computes surface-node normals and nodal area weights from surface-face connectivity.

It also prepares reusable topology indices so that coordinate-independent connectivity processing does not need to be repeated for each deformation.

### `rotation.py`

Computes local rotation matrices and translation vectors from the original and deformed surface geometry.

### `symmetry.py`

Provides reusable symmetry-plane geometry operations, including:

* point projection,
* point and vector reflection,
* symmetry-plane point detection, and
* approximate symmetry correction.

### `derivative.py`

Provides JAX automatic-differentiation utilities for:

* Jacobian-vector products,
* vector-Jacobian products,
* finite-difference comparison, and
* JVP–VJP dot-product consistency checks.

## Requirements

The core implementation requires:

* Python 3.10 or newer
* JAX
* JAX NumPy

Install JAX according to the official JAX installation instructions for the intended CPU or GPU platform.

For a basic CPU installation:

```bash
pip install jax
```

GPU installation depends on the installed CUDA version and should follow the official JAX documentation.

## Public Interface

### `deform_mesh`

```python
from driver import deform_mesh
```

`deform_mesh(...)` performs one complete mesh-deformation calculation.

Its main inputs are:

| Input                  |            Shape | Description                                                    |
| ---------------------- | ---------------: | -------------------------------------------------------------- |
| `Xv0`                  |  `(n_volume, 3)` | Original volume-mesh coordinates                               |
| `Xs0`                  | `(n_surface, 3)` | Original surface-mesh coordinates                              |
| `surface_displacement` | `(n_surface, 3)` | Surface displacement measured from `Xs0`                       |
| `conn`                 |      `(n_conn,)` | Flattened surface-face connectivity                            |
| `face_sizes`           |     `(n_faces,)` | Number of nodes in each surface face                           |
| `surface_global_ids`   |   `(n_surface,)` | Mapping from local surface-node indices to volume-node indices |

The returned value is a JAX array with shape `(n_volume, 3)` containing the deformed volume coordinates.

The caller is responsible for supplying:

* corresponding volume and surface arrays,
* consistent floating-point dtypes,
* valid fixed-topology connectivity, and
* valid surface-to-volume point indices.

The public interface does not perform file I/O or OpenFOAM-specific processing.

### `make_deformation_function`

```python
from driver import make_deformation_function
```

`make_deformation_function(...)` prepares and captures the fixed mesh arrays,
reference geometry, topology indices, and configuration values, then returns
a one-input function:

```text
surface_displacement -> deformed volume coordinates
```

The prepared data is reused across subsequent deformation evaluations. This
form is intended for repeated calculations and JAX transformations such as
JVP and VJP.

## Minimal Example

```python
import jax.numpy as jnp

from driver import deform_mesh

Xv0 = jnp.asarray(volume_points)
Xs0 = jnp.asarray(surface_points)

surface_displacement = jnp.asarray(
    target_surface_points - surface_points
)

conn = jnp.asarray(
    surface_connectivity,
    dtype=jnp.int32,
)

face_sizes = jnp.asarray(
    surface_face_sizes,
    dtype=jnp.int32,
)

surface_global_ids = jnp.asarray(
    surface_to_volume_ids,
    dtype=jnp.int32,
)

Xv = deform_mesh(
    Xv0=Xv0,
    Xs0=Xs0,
    surface_displacement=surface_displacement,
    conn=conn,
    face_sizes=face_sizes,
    surface_global_ids=surface_global_ids,
    symmetry_mode="nonesym",
)
```

## Symmetry Modes

The supported symmetry modes are:

| Mode        | Description                                                                                  |
| ----------- | -------------------------------------------------------------------------------------------- |
| `nonesym`   | Uses only the supplied surface without symmetry treatment                                    |
| `exactsym`  | Includes mirrored contributions directly in the IDW interpolation                            |
| `approxsym` | Performs one-sided IDW and then suppresses plane-normal displacement near the symmetry plane |

### No Symmetry

```python
Xv = deform_mesh(
    Xv0=Xv0,
    Xs0=Xs0,
    surface_displacement=surface_displacement,
    conn=conn,
    face_sizes=face_sizes,
    surface_global_ids=surface_global_ids,
    symmetry_mode="nonesym",
)
```

### Exact Symmetry

Exact symmetry evaluates both the real-side and reflected-side contributions during IDW interpolation.

```python
Xv = deform_mesh(
    Xv0=Xv0,
    Xs0=Xs0,
    surface_displacement=surface_displacement,
    conn=conn,
    face_sizes=face_sizes,
    surface_global_ids=surface_global_ids,
    symmetry_mode="exactsym",
    symmetry_plane_point=jnp.array([0.0, 0.0, 0.0]),
    symmetry_plane_normal=jnp.array([0.0, 0.0, 1.0]),
)
```

The symmetry plane is defined by:

```text
dot(x - symmetry_plane_point, symmetry_plane_normal) = 0
```

### Approximate Symmetry

Approximate symmetry performs a one-sided IDW deformation and then reduces the displacement component normal to the symmetry plane.

The normal displacement is zero on the symmetry plane and recovers linearly to its full value over `symmetry_length_scale`.

```python
Xv = deform_mesh(
    Xv0=Xv0,
    Xs0=Xs0,
    surface_displacement=surface_displacement,
    conn=conn,
    face_sizes=face_sizes,
    surface_global_ids=surface_global_ids,
    symmetry_mode="approxsym",
    symmetry_plane_point=jnp.array([0.0, 0.0, 0.0]),
    symmetry_plane_normal=jnp.array([0.0, 0.0, 1.0]),
    symmetry_length_scale=1.0,
)
```

> [!CAUTION]
> `approxsym` is an experimental approximation. Its behavior should be validated for each mesh, deformation, and application before use in production calculations.

## Derivative Usage

The derivative utilities operate on a function that maps surface displacement to deformed volume coordinates.

First create that function:

```python
from driver import make_deformation_function

deformation_function = make_deformation_function(
    Xv0=Xv0,
    Xs0=Xs0,
    conn=conn,
    face_sizes=face_sizes,
    surface_global_ids=surface_global_ids,
    symmetry_mode="nonesym",
)
```

### Jacobian-Vector Product

A Jacobian-vector product evaluates:

```text
J @ surface_direction
```

where `J` is the Jacobian of the volume coordinates with respect to the surface displacement.

```python
import jax.numpy as jnp

from derivative import mesh_deformation_jvp

surface_direction = jnp.ones_like(
    surface_displacement
)

Xv, volume_direction = mesh_deformation_jvp(
    deformation_function=deformation_function,
    surface_displacement=surface_displacement,
    surface_direction=surface_direction,
)
```

`volume_direction` has the same shape as the volume coordinates.

### Vector-Jacobian Product

A vector-Jacobian product evaluates:

```text
J.T @ volume_seed
```

```python
from derivative import mesh_deformation_vjp

volume_seed = jnp.ones_like(Xv0)

Xv, surface_sensitivity = mesh_deformation_vjp(
    deformation_function=deformation_function,
    surface_displacement=surface_displacement,
    volume_seed=volume_seed,
)
```

`surface_sensitivity` has the same shape as the surface displacement.

### Derivative Checks

Two preliminary consistency checks are provided:

```python
from derivative import (
    check_jvp_with_finite_difference,
    check_jvp_vjp_dot_product,
)
```

`check_jvp_with_finite_difference(...)` compares the JAX JVP with a centered finite-difference approximation.

`check_jvp_vjp_dot_product(...)` checks the identity:

```text
volume_seed · (J @ surface_direction)
=
(J.T @ volume_seed) · surface_direction
```

> [!WARNING]
> The derivative implementation and derivative checks are currently under development. They have not yet been fully tested or validated on practical mesh-deformation cases. Correctness is therefore not guaranteed.

## Numerical Parameters

The main optional parameters are:

| Parameter            |   Default | Description                                               |
| -------------------- | --------: | --------------------------------------------------------- |
| `Ldef`               |     `1.0` | Reference length used by the IDW weights                  |
| `aExp`               |     `3.0` | First IDW exponent                                        |
| `bExp`               |     `5.0` | Second IDW exponent                                       |
| `alpha`              |    `0.25` | Relative scaling of the second IDW term                   |
| `normal_eps`         | `1.0e-30` | Regularization used by normal calculations                |
| `rotation_eps`       | `1.0e-12` | Tolerance used by rotation calculations                   |
| `warp_eps`           | `1.0e-30` | Regularization used by IDW and symmetry operations        |
| `volume_chunk_size`  |     `512` | Number of volume points processed per Python-level chunk  |
| `surface_block_size` |    `1024` | Number of surface points processed per compiled JAX block |

Appropriate regularization values may depend on whether the calculation uses single or double precision.

## JAX Behavior

The mesh-deformation functions return JAX arrays.

The public interface intentionally does not call:

```python
block_until_ready()
jax.device_get()
```

This allows the returned calculation to remain compatible with JAX transformations and asynchronous execution.

Callers that need timing, NumPy conversion, or file output should synchronize explicitly:

```python
Xv.block_until_ready()
Xv_numpy = np.asarray(jax.device_get(Xv))
```

## Current Development Status

The project is under active development.

Current status:

* `nonesym`: implemented
* `exactsym`: implemented
* `approxsym`: experimental and still under development
* JVP and VJP utilities: implemented but not fully validated
* derivative consistency checks: implemented but not fully validated

Future revisions may improve performance, testing, examples, validation, and
documentation.


## References
This implementation is inspired by the mesh deformation method used in IDWarp.

### Paper
- Secco, Ney R. and Kenway, Gaetan K. W. and He, Ping and Mader, Charles and Martins, Joaquim R. R. A.  
  "Efficient Mesh Generation and Deformation for Aerodynamic Shape Optimization,"  
  AIAA Journal, vol. 59, no. 4, pp. 1151–1168, 2021.  
  [https://doi.org/10.2514/1.J059491](https://doi.org/10.2514/1.J059491)

### Related Software
- [IDWarp GitHub repository](https://github.com/mdolab/idwarp.git)




