import numpy as np
from time import perf_counter

from .base_tb import TightBinding
from ...geometry import Geometry


class TightBindingKMIsland(TightBinding):
    """
    Kane–Mele model in real space on a finite honeycomb island (open boundaries).

    Basis per site i:
      |i, up>, |i, down>

    Terms:
      1) NN hopping: t * sum_<ij>,s c†_{is} c_{js}
      2) Intrinsic SOC (KM): i*lam_so * sum_<<ij>> nu_ij c†_i s_z c_j
      3) Optional mass term: m * sum_i xi_i c†_i c_i, with xi_i = +1 (A), -1 (B)
    """

    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "island_km"
        self.H = None
        self.E = None
        self.U = None
        self.band_structure_data = None

        # If you later add these to ModelOptions, getattr will pick them up.
        self.t = getattr(model_options, "t", -1.0)
        self.lam_so = getattr(model_options, "lam_so", 0.1)
        self.mass = getattr(model_options, "mass", 0.0)

    @staticmethod
    def _iup(i: int) -> int:
        return 2 * i

    @staticmethod
    def _idn(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _nu_ij(pos_i, pos_k, pos_j) -> int:
        """
        Sign of the out-of-plane cross product for the turn i -> k -> j.
        Returns +1 or -1. If degenerate, returns 0.
        """
        d_ik = pos_k - pos_i
        d_kj = pos_j - pos_k
        cross_z = d_ik[0] * d_kj[1] - d_ik[1] * d_kj[0]
        if np.isclose(cross_z, 0.0):
            return 0
        return 1 if cross_z > 0 else -1

    def build_hamiltonian(self, geometry: Geometry):
        print("Building 'KM Island' Hamiltonian...")

        pos = np.asarray(geometry.sites, dtype=float)  # (N,2)
        N = pos.shape[0]
        dim = 2 * N
        H = np.zeros((dim, dim), dtype=complex)

        Cnn = geometry.nn_connectivity_matrix.astype(int)
        Cnnn = geometry.nnn_connectivity_matrix.astype(int)

        # 1) NN hopping (spin-independent)
        ii, jj = np.where(Cnn == 1)
        for i, j in zip(ii, jj):
            if i == j:
                continue
            H[self._iup(i), self._iup(j)] += self.t
            H[self._idn(i), self._idn(j)] += self.t

        # 2) Intrinsic SOC on NNN bonds
        if self.lam_so != 0.0:
            # For each NNN pair (i,j), find common NN intermediates k with i-k and k-j NN
            nnii, nnjj = np.where(Cnnn == 1)
            for i, j in zip(nnii, nnjj):
                if i == j:
                    continue

                common_k = np.where((Cnn[i] == 1) & (Cnn[:, j] == 1))[0]
                if common_k.size == 0:
                    continue

                # Sum over all available 2-step paths (important at edges where one path may be missing)
                nu_sum = 0
                for k in common_k:
                    nu_sum += self._nu_ij(pos[i], pos[k], pos[j])

                if nu_sum == 0:
                    continue

                # For honeycomb bulk, nu_sum will be ±2 (two paths), at edges often ±1 (one path)
                nu_ij = nu_sum

                H[self._iup(i), self._iup(j)] += 1j * self.lam_so * nu_ij
                H[self._idn(i), self._idn(j)] += -1j * self.lam_so * nu_ij

        # 3) Optional staggered sublattice potential (mass term)
        if self.mass != 0.0:
            if not hasattr(geometry, "sublattice_label_idxs"):
                raise AttributeError("Geometry must provide sublattice_label_idxs for mass term.")

            sub = np.asarray(geometry.sublattice_label_idxs, dtype=int)
            if sub.shape[0] != N:
                raise ValueError("geometry.sublattice_label_idxs must have length N.")

            # honeycomb: A=0, B=1
            xi = np.where(sub == 0, +1.0, -1.0)

            for i in range(N):
                H[self._iup(i), self._iup(i)] += self.mass * xi[i]
                H[self._idn(i), self._idn(i)] += self.mass * xi[i]

        # Enforce Hermiticity
        H = 0.5 * (H + H.conj().T)

        self.H = H
        print("'KM Island' Hamiltonian Done.")

    def solve_eigenvalues(self, geometry: Geometry, H_type="real"):
        assert H_type == "real", "KM island is real-space only (no k)."
        if self.H is None:
            raise RuntimeError("Hamiltonian not built. Call build_hamiltonian first.")

        print("Calculating 'KM Island' Eigenvalues...")
        start = perf_counter()
        self.E, self.U = np.linalg.eigh(self.H)
        print("'KM Island' Eigenvalues Done.")
        return perf_counter() - start

    def build_band_structure(self, geometry: Geometry):
        self.band_structure_data = None
