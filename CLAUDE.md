# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

pyFES — Python Finite Element Solver for electromagnetics.

## Commands

- Run all tests: `python -m pytest tests/ -v`
- Run a single test: `python -m pytest tests/test_core.py::TestShapeFunctions::test_2d_linear -v`
- Install in editable mode: `python -m pip install -e .`
- Activate venv: `source .venv/bin/activate` (or `python -m pytest`) — venv is at `$CWD/.venv/`

## Architecture

Package structure:

- `pyfes/constants.py` — Physical constants (c0, z0, eps0, mu0) and utility functions (dB conversion, phase unwrap)
- `pyfes/mesh/` — Mesh generation, I/O, and visualization
  - `io_poly.py` — Read Triangle .poly meshes, write .poly geometry files
  - `build.py` — Generate regular triangular meshes on the unit square
  - `plot.py` — Matplotlib-based mesh and geometry plotting
- `pyfes/fem/` — Core finite element method routines
  - `shape_functions.py` — Scalar Lagrange basis on intervals and triangles (orders 1-4), H(curl)-conforming vector basis, DOF counting per element
  - `quadrature.py` — Gauss-Legendre quadrature on [0,1] and Duffy-transformed simplex quadrature for triangles
  - `jacobian.py` — Jacobian determinant and inverse transpose for triangular elements
  - `dof.py` — Global DOF numbering and physical coordinate positioning, including hierarchical basis function indices for high-order elements
  - `boundary.py` — Boundary DOF mapping for domain decomposition (DD) methods
  - `assembly.py` — System matrix assembly (linear, waveguide ports, domain decomposition), boundary condition handling (ABC, Dirichlet, Neumann, DD, waveguide ports), Helmholtz/scalar Laplacian operators
- `pyfes/post/plot.py` — pyVista-based in-process rendering: `plot_field()`, `plot_electric_field()`, `plot_magnetic_field()`, `plot_mesh()`

## Key conventions

- Mesh uses 0-based indexing throughout (MATLAB used 1-based)
- Sparse matrices use scipy.sparse.csr_matrix
- Shape functions are returned as lambda functions that evaluate at reference coordinates
- The reference triangle has vertices (0,0), (1,0), (0,1)
- `sys` dict carries all system state (solver config, assembled matrices, solution)
- `mesh` dict carries all mesh geometry and topology
