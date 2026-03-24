Real-Space Topological States in Finite Honeycomb Islands
========================================================

This repository contains the full codebase and analysis developed for my MSci Physics dissertation at University College London.

The project investigates boundary, defect, and bulk physics in finite (0D) honeycomb island lattices using real-space tight-binding models. The primary observable is the local density of states (LDOS), analysed across three Hamiltonian families:

- Nearest-neighbour (NN) honeycomb model (control case)
- Kane–Mele (KM) model with intrinsic spin–orbit coupling
- Acceptor-inspired multi-orbital model with direction-dependent hopping

The aim is to identify experimentally relevant signatures of edge states, defect-bound states, and bulk modes in finite systems, motivated by STM measurements of atomically engineered lattices.

---

Key Features
------------

- Finite honeycomb island construction (shared geometry)
- Vacancy modelling:
  
  - bulk vacancy
  - edge vacancy
  - continuous vacancy-position sweeps

- LDOS computation with controlled broadening
- Region-projected LDOS (edge, bulk, defect)
- Peak attribution linking LDOS features to eigenstates
- Comparative analysis across NN, KM, and acceptor models

---

Repository Structure
--------------------

Notebooks::

    0d_LDOS.ipynb
    0d_acceptor_LDOS.ipynb
    0dkm_LDOS.ipynb
    0d_plots.ipynb
    0d_island.ipynb
    0d_island_setup.ipynb
    0dkm_island.ipynb

Core source code (src/topological_insulator/python/)::

    geometry/geometry.py
    cell_parser/
    hamiltonian/tight_binding/
        base_tb.py
        bulk_tb.py
        edge_tb.py
        island_tb.py
        islandkm_tb.py
        island_acceptor_tb.py
    topological_invariants/

All figures are generated within the notebooks.

---

How to Run
----------

1. Install standard Python packages:

   - numpy
   - scipy
   - matplotlib
   - jupyter

2. Run notebooks in sequence:

   - 0d_island_setup.ipynb → geometry construction
   - 0d_LDOS.ipynb → NN model
   - 0dkm_LDOS.ipynb → KM model
   - 0d_acceptor_LDOS.ipynb → acceptor model
   - 0d_plots.ipynb → final figures

The repository is fully self-contained.

---

Acknowledgements and Code Provenance
------------------------------------

This repository builds on an existing tight-binding framework developed in part by Javier Larrain Garcia-Perate and collaborators.

Within the ``topological_insulator`` module, the following core files are derived from this prior framework:

- ``base_tb.py``
- ``bulk_tb.py``
- ``edge_tb.py``

All other files in this repository were developed or modified as part of this MSci dissertation project, including:

Notebooks:

- ``0d_LDOS.ipynb``
- ``0d_LDOSold.ipynb``
- ``0d_acceptor_LDOS.ipynb``
- ``0d_island.ipynb``
- ``0d_island_setup.ipynb``
- ``0d_plots.ipynb``
- ``0dkm_LDOS.ipynb``
- ``0dkm_island.ipynb``

Tight-binding implementations:

- ``island_tb.py``
- ``islandkm_tb.py``
- ``island_acceptor_tb.py``

Geometry and supporting modules:

- ``geometry.py``
- modules within ``cell_parser/``, ``geometry/``, and ``topological_invariants/`` (adapted and extended where required)

These components implement the full real-space LDOS analysis pipeline developed in this project, including:

- finite island construction
- vacancy modelling (bulk, edge, and sweeps)
- LDOS computation and projection
- peak attribution and eigenstate analysis

The full repository is included to ensure that the codebase is self-contained and fully reproducible, so that all notebooks and workflows can run without requiring any additional files beyond standard Python dependencies.

---

Notes
-----

- All results correspond to finite (0D) island geometries with open boundaries.
- The analysis is designed to be directly comparable to STM-style LDOS measurements.
- This repository reflects the final state of the dissertation code and figures.
