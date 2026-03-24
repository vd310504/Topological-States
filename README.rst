## Acknowledgements and Code Provenance

This repository builds on an existing tight-binding framework developed by Javier Larrain Garcia-Perate and other LCN collaborators.

All other files in this repository were developed or modified as part of this MSci dissertation project, including:

### Notebooks
- `0d_LDOS.ipynb`
- `0d_LDOSold.ipynb`
- `0d_acceptor_LDOS.ipynb`
- `0d_island.ipynb`
- `0d_island_setup.ipynb`
- `0d_plots.ipynb`
- `0dkm_LDOS.ipynb`
- `0dkm_island.ipynb`

### Tight-binding and model implementations
- `island_tb.py`
- `islandkm_tb.py`
- `island_acceptor_tb.py`

### Geometry and supporting code
- `geometry.py`
- all supporting modules within `cell_parser/`, `geometry/`, and `topological_invariants/` (adapted and extended where required for this project)

These components implement the full real-space LDOS analysis pipeline developed in this work, including:

- finite island construction  
- vacancy modelling (bulk, edge, and positional sweeps)  
- LDOS computation and projection  
- peak attribution and eigenstate analysis  

The full repository is included to ensure that the codebase is self-contained and fully reproducible, so that all notebooks and analysis workflows can be executed without requiring any additional files beyond standard Python dependencies.

---

## Notes

- All results correspond to finite (0D) island geometries with open boundaries.
- The analysis is designed to be directly comparable to STM-style LDOS measurements.
- The repository reflects the final state of the dissertation code and figures.

---

## How to Use

1. Install standard scientific Python packages:
   - numpy
   - scipy
   - matplotlib
   - jupyter

2. Run notebooks in order:
   - `0d_island_setup.ipynb` → builds geometry
   - `0d_LDOS.ipynb` → NN analysis
   - `0dkm_LDOS.ipynb` → KM analysis
   - `0d_acceptor_LDOS.ipynb` → acceptor analysis
   - `0d_plots.ipynb` → final figures

All analysis is self-contained within the repository.

