# rslattice
Python module for doing calculations on integer lattices.

Currently implemented:
 * [LLL reduction](https://en.wikipedia.org/wiki/Lenstra%E2%80%93Lenstra%E2%80%93Lov%C3%A1sz_lattice_basis_reduction_algorithm) with custom bilinear form
 * [Hermite normal form](https://en.wikipedia.org/wiki/Hermite_normal_form)
 * Babai's nearest plane algorithm for solving approximate [CVP](https://en.wikipedia.org/wiki/Lattice_problem#Closest_vector_problem_(CVP))
 * Exact integer determinant using [Bareiss algorithm](https://en.wikipedia.org/wiki/Bareiss_algorithm)

Implemented in Rust because we gotta go FAST.
This is mainly intended to speed up https://github.com/Sin-tel/temper/, but it should be generally useful.
