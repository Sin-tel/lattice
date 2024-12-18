# rslattice
Python module for doing calculations on integer lattices.

Currently implemented:
 * [LLL reduction](https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm) with custom bilinear form
 * [Hermite normal form](https://en.wikipedia.org/wiki/Hermite_normal_form)
 * Babai's nearest plane algorithm for solving approximate [CVP](https://en.wikipedia.org/wiki/Lattice_problem#Closest_vector_problem_(CVP))
 * Exact integer determinant using [Bareiss algorithm](https://en.wikipedia.org/wiki/Bareiss_algorithm)

Implemented in Rust because we gotta go FAST.
This is mainly intended to speed up https://github.com/Sin-tel/temper/, but it should be generally useful.

## Installation
```bash
pip install rslattice
```

## Usage

 * `rslattice.hnf(basis)`
   - `basis`: ndarray int64 (r, n)
   - returns: ndarray int64 (r, n)

   Calculates row-style hermite normal form of `basis`. The calculation uses BigInt internally, but returns int64 so it may raise an OverflowException.

 * `rslattice.lll(basis, delta, w)`
   - `basis`: ndarray int64 (r, n)
   - `delta`: float
   - `w`: ndarray float64 (n, n)
   - returns: ndarray int64 (r, n)

   Performs LLL reduction of basis. Large values of delta lead to stronger reductions of the basis. A good starting point is `delta=0.75`.
   `w` is a square matrix such that the inner product is $\left\langle a, b \right\rangle = a^{\mathsf{T}} W b$.
   Set it to the identity matrix (`np.eye(n)`) if you don't need a custom inner product.
   Assumes the basis is linearly independent.

 * `rslattice.nearest_plane(v, basis, w)`
   - `v`: ndarray int64 (n)
   - `basis`: ndarray int64 (r, n)
   - `w`: ndarray float64 (n, n)
   - returns: ndarray int64

   Solve approximate closest vector problem using Babai's nearest plane algorithm.
   That is, it returns a vector that is in the row span of `basis` that is close to `v`.
   The `w` argument is used as in LLL above.
   Assumes the basis is linearly independent.

 * `rslattice.integer_det(basis)`
   - `v`: ndarray int64 (n, n)
   - returns: int

   Calculates exact integer determinant. May raise an OverflowException.


