"""
Fractional Brownian Motion (fBm) Generator
============================================

Generates synthetic fBm trajectories with known Hurst exponents for use as
training data in the Hurst-exponent-prediction project.

Three generation methods are implemented:
    - "cholesky"     : Exact, O(T^2) memory/time. Good for small T or as a
                        ground-truth reference.
    - "davies-harte" : Exact (when it works) and fast, O(T log T) via FFT
                        circulant embedding. Recommended for large-scale
                        dataset generation.
    - "hosking"      : Recursive conditional method, O(T^2) but numerically
                        stable and doesn't require the circulant matrix to be
                        positive semi-definite.

All three are exact simulators of fBm (unlike, e.g., the random midpoint
displacement method, which is only approximate).
"""


import numpy as np

# note there is a python moduel to generate fgn.
# here, not using


class FractionalBrownianMotionGenerator:
    """Generate fractional Brownian motion (fBm) trajectories.

    Parameters
    ----------
    T : int
        Number of time steps in each generated trajectory (path length).
    dt : float, default=1.0
        Time increment between consecutive samples.
    method : {"davies-harte", "cholesky", "hosking"}, default="davies-harte"
        Simulation method used by ``generate_path``.
    seed : int, optional
        Seed for the internal NumPy random generator (for reproducibility).
    """

    def __init__(self, 
                 T: int, 
                 dt: float = 1.0, 
                 method: str = "davies-harte",
                 seed: int | None = None):
        if T < 2:
            raise ValueError("T must be >= 2")
        if method not in ("davies-harte", "cholesky", "hosking"):
            raise ValueError(f"Unknown method: {method}")

        self.T = T
        self.dt = dt
        self.method = method
        self.rng = np.random.default_rng(seed)

    def autocovariance(self, 
                       H: float, 
                       lags: np.ndarray | None = None) -> np.ndarray:
        """
        This is the increment process of fBm; fGn is stationary with this
        autocovariance.

        Parameters
        ----------
        H : float
            Hurst exponent in (0, 1).
        lags : array-like of int, optional
            Lags at which to evaluate gamma. Defaults to 0..T-1.
        """
        if lags is None:
            lags = np.arange(self.T)
        lags= np.asarray(lags, dtype=float)
        gamma = 0.5 * (
            np.abs(lags - 1) ** (2 * H)
              - 2 * np.abs(lags) ** (2 * H)
            + np.abs(lags + 1) ** (2 * H)
        )
        return gamma

    # ------------------------------------------------------------------
    # Simulation methods
    # ------------------------------------------------------------------
    def _fgn_cholesky(self, H: float) -> np.ndarray:

        """Cholesky factorization"""

        n = self.T
        idx = np.arange(n)
        gamma = self.autocovariance(H, idx)
        cov = gamma[np.abs(idx[:, None] - idx[None, :])]
        L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
        z = self.rng.standard_normal(n)
        fgn = L @ z
        return fgn

    def _fgn_davies_harte(self, H: float) -> np.ndarray:
        """Davies & Harte, 1987).
        """
        n = self.T
        m = 2 * (n - 1)
        k = np.arange(m)




        gamma = self.autocovariance(H, np.arange(n))
        row = np.concatenate([gamma, gamma[-2:0:-1]])
        eigvals = np.fft.fft(row).real

        if np.min(eigvals) < -1e-9:

            return self._fgn_hosking(H)

        eigvals = np.maximum(eigvals, 0)  
        

        Zr = self.rng.standard_normal(m)
        Zi = self.rng.standard_normal(m)
        Z = Zr + 1j * Zi

        W = np.fft.fft(np.sqrt(eigvals / m) * Z)
        fgn = W[:n].real
        return fgn


    def _fgn_hosking(self, H: float) -> np.ndarray:
        """Exact simulation via Hosking's recursive method.

            Very good for numerical stability but Very expansive, time wise
        """
        n = self.T
        gamma = self.autocovariance(H, np.arange(n))

        fgn = np.zeros(n)
        fgn[0] = self.rng.standard_normal() * np.sqrt(gamma[0])

        phi = np.zeros(n)
        psi = np.zeros(n)
        v = gamma[0]

        phi[0] = gamma[1] / gamma[0]
        fgn[1] = phi[0] * fgn[0] + np.sqrt(v * (1 - phi[0] ** 2)) * self.rng.standard_normal()
        v *= (1 - phi[0] ** 2)
        psi[0] = phi[0]

        for i in range(2, n):
            num = gamma[i] - np.sum(psi[:i] * gamma[1:i + 1][::-1])
            phi_i = num / v
            phi_new = psi[:i] - phi_i * psi[:i][::-1]
            psi[:i] = phi_new
            psi[i - 1] = phi_i
            v *= (1 - phi_i ** 2)
            mean = np.sum(psi[:i] * fgn[i - 1::-1])
            fgn[i] = mean + np.sqrt(max(v, 0)) * self.rng.standard_normal()

        return fgn


    def _generate_fgn(self, H: float) -> np.ndarray:

        if self.method == "cholesky":
            fgn = self._fgn_cholesky(H)

        elif self.method == "davies-harte":
            fgn = self._fgn_davies_harte(H)

        else:
            fgn = self._fgn_hosking(H)
        return fgn

    def generate_path(self, 
                      H: float, 
                      include_start: bool = True) -> np.ndarray:
        """
        Single traj

        Parameters
        ----------
        H : Hurst exponent in (0, 1).
        
        include_start : default=True         (start with a zero)
        """
        if not (0.0 < H < 1.0):
            raise ValueError("H must be between 0 and 1")

        fgn = self._generate_fgn(H)
        scale = self.dt ** H
        fbm = np.cumsum(fgn) * scale

        if include_start:
            fbm = np.concatenate([[0.0], fbm])[: self.T]

            if len(fbm) < self.T:
                fbm = np.concatenate([fbm, np.zeros(self.T - len(fbm))])


        return fbm


# many traejctories
    def generate_dataset(self, 
                         n_samples: int, 
                         H_range: tuple[float, float] = (0.1, 0.9),
                         H_values: np.ndarray | None = None):
        
        """Generate a full dataset

        Parameters
        ----------
        n_samples : int
            Number of trajectories to generate.
        H_range : (float, float), default=(0.1, 0.9)
            Range from which H values are drawn uniformly at random (used
            only if ``H_values`` is not supplied).
        H_values : array-like, optional
            Explicit array of H values (length n_samples) to use instead of
            random sampling from H_range. Useful for stratified / uniform
            grid sampling.

        Returns
        -------
        X : np.ndarray, shape (n_samples, T)
            Generated trajectories.
        y : np.ndarray, shape (n_samples,)
            Corresponding Hurst exponents.
        """
        if H_values is None:
            H_values = self.rng.uniform(H_range[0], H_range[1], size=n_samples)
        else:
            H_values = np.asarray(H_values)
            assert len(H_values) == n_samples, "H_values must have length n_samples"

        X = np.zeros((n_samples, self.T))
        for i, H in enumerate(H_values):
            X[i] = self.generate_path(float(H))

        return X, H_values.astype(np.float64)


if __name__ == "__main__":

    gen = FractionalBrownianMotionGenerator(T=512, 
                                            method="davies-harte", 
                                            seed=1234)

    X, y = gen.generate_dataset(n_samples=5, 
                                H_range=(0.1, 0.9))

    print("X shape:", X.shape, "y:", y)
