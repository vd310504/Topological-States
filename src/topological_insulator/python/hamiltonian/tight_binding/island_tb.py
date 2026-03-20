import numpy as np
from time import perf_counter

from .base_tb import TightBinding
from ...geometry import Geometry


class TightBindingIsland(TightBinding):
    """
    0D island tight-binding:
    - builds ONE real-space Hamiltonian H (no k-space, no Bloch phases)
    - diagonalises once, discrete spectrum
    - this uses a graphene-style (1 orbital per site) using NN connectivity
    """

    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "island"
        self.H = None
        self.E = None
        self.U = None
        self.band_structure_data = None

    def build_hamiltonian(self, geometry: Geometry):
        print("Building 'Island' Hamiltonian...")

        # simplest graphene-like hopping t (scalar)
        # honeycomb.json layout: eigenvalues.A.nn_hopping.B.t_ss_sigma
        # evA = getattr(self.cell_parser.eigenvalues, "A").value
        # t = evA["nn_hopping"]["B"]["t_ss_sigma"]

        # Forcing graphene-like hopping for island TB
        t = -1.0

        N = len(geometry.sites)
        H = np.zeros((N, N), dtype=float)

        # Use NN connectivity: C[i,j] = 1 means NN bond
        C = geometry.nn_connectivity_matrix
        H[C == 1] = t

        # Force symmetry/Hermiticity
        H = 0.5 * (H + H.T)

        self.H = H
        print("'Island' Hamiltonian Done.")

    def solve_eigenvalues(self, geometry: Geometry, H_type="real"):
        assert H_type == "real", "Island is real-space only (no k)."
        if self.H is None:
            raise RuntimeError("Island Hamiltonian not built. Call build_hamiltonian first.")

        print("Calculating 'Island' Eigenvalues...")
        start = perf_counter()

        self.E, self.U = np.linalg.eigh(self.H)

        print("'Island' Eigenvalues Done")
        return perf_counter() - start

    def build_band_structure(self, geometry: Geometry):
        # Not applicable for 0D systems (no k-path)
        self.band_structure_data = None