# rslattice
Python module for doing calculations on integer lattices.

Currently implemented:
 * LLL reduction with custom bilinear form
 * Hermite normal form

Implemented in Rust because we gotta go FAST.
This is mainly intended to speed up https://github.com/Sin-tel/temper/, but it should be generally useful.
