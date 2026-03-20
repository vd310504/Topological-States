import numpy as np
from time import perf_counter

from .base_tb import TightBinding
from ...geometry import Geometry


class TightBindingAcceptorIsland(TightBinding):
    """
    Acceptor-based tight-binding model in real space on a finite honeycomb island (open boundaries).

    Basis per site i (spinless here):
      |i, a0>, |i, a1>, |i, a2>, |i, a3>    (4 acceptor orbitals)

    Hamiltonian dimension: 4N

    Terms (minimal version):
      1) NN hopping: sum_<ij> c†_i  T(dir_ij)  c_j
         where T(dir) is a 4x4 matrix depending on bond orientation.
         You provide 3 matrices corresponding to the three honeycomb NN directions:
            T0   : "0 degrees" (your quantisation-direction hopping)
            T120 : +120 degrees
            Tm120: -120 degrees

      2) Optional onsite term: H_onsite (4x4) applied to every site.
    """

    def __init__(self, model_options, cell_parser):
        super().__init__(model_options, cell_parser)
        self.location = "island_acceptor"
        self.H = None
        self.E = None
        self.U = None
        self.band_structure_data = None

        # Parameters / matrices
        self.T0 = None        # 4x4
        self.T120 = None      # 4x4
        self.Tm120 = None     # 4x4

        # Optional onsite 4x4 term (default: zero)
        self.H_onsite = np.zeros((4, 4), dtype=complex)

        # Angle classification tolerance (radians)
        self.angle_tol = 0.2 # ~11 degrees, adjust if geometry is rotated
    @staticmethod
    def _wrap_angle(theta):
        """Wrap angle to (-pi, pi]."""
        return (theta + np.pi) % (2 * np.pi) - np.pi

    def _pick_T(self, dx, dy):
        """
        Pick which 4x4 hopping matrix to use based on the bond angle.
        We compare the bond angle to {0, +2pi/3, -2pi/3} (up to wrapping).
        """
        if self.T0 is None or self.T120 is None or self.Tm120 is None:
            raise RuntimeError("Set T0, T120, Tm120 (all 4x4) before building the Hamiltonian.")

        theta = np.arctan2(dy, dx)  # (-pi, pi]
        candidates = np.array([0.0, np.pi/3, -np.pi/3], dtype=float)  # {0, +60°, -60°} offset to make it correspond to the 3 NN directions in the honeycomb lattice.
        diffs = np.abs(np.array([self._wrap_angle(theta - a) for a in candidates]))

        k = int(np.argmin(diffs))
        if diffs[k] > self.angle_tol:
            if not hasattr(self, "_warned_angle_tol"):
                print("Warning: bond angle not close to {0, ±120} deg within angle_tol.")
                print("         Consider increasing angle_tol or updating _pick_T convention.")
                self._warned_angle_tol = True

        if k == 0:
            return self.T0
        elif k == 1:
            return self.T120
        else:
            return self.Tm120

    def build_hamiltonian(self, geometry: Geometry):
        print("Building 'Acceptor Island' Hamiltonian...")

        pos = np.asarray(geometry.sites, dtype=float)  # (N,2)
        N = pos.shape[0]
        dim = 4 * N
        H = np.zeros((dim, dim), dtype=complex)

        # Optional onsite block on each site
        if self.H_onsite is not None:
            Hons = np.asarray(self.H_onsite, dtype=complex)
            if Hons.shape != (4, 4):
                raise ValueError("H_onsite must be a 4x4 matrix.")
            for i in range(N):
                si = slice(4*i, 4*i + 4)
                H[si, si] += Hons

        # NN connectivity
        Cnn = geometry.nn_connectivity_matrix.astype(int)

        # Fill hopping once per undirected bond (i<j), then add Hermitian conjugate
        ii, jj = np.where(Cnn == 1)
        for i, j in zip(ii, jj):
            if i == j:
                continue
            if j < i:
                continue

            dx, dy = pos[j] - pos[i]
            Tij = self._pick_T(dx, dy)  # 4x4

            Tij = np.asarray(Tij, dtype=complex)
            if Tij.shape != (4, 4):
                raise ValueError("Each hopping matrix (T0/T120/Tm120) must be 4x4.")

            si = slice(4*i, 4*i + 4)
            sj = slice(4*j, 4*j + 4)

            H[si, sj] += Tij
            H[sj, si] += Tij.conj().T

        # Enforce Hermiticity
        H = 0.5 * (H + H.conj().T)

        self.H = H
        print("'Acceptor Island' Hamiltonian Done.")

    def solve_eigenvalues(self, geometry: Geometry, H_type="real"):
        assert H_type == "real", "Acceptor island is real-space only (no k)."
        if self.H is None:
            raise RuntimeError("Hamiltonian not built. Call build_hamiltonian first.")

        print("Calculating 'Acceptor Island' Eigenvalues...")
        start = perf_counter()
        self.E, self.U = np.linalg.eigh(self.H)
        print("'Acceptor Island' Eigenvalues Done.")
        return perf_counter() - start

    def build_band_structure(self, geometry: Geometry):
        self.band_structure_data = None
