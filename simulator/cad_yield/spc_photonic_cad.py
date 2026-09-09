"""Photonic CAD, DCE squeezing, electro-optic conversion and SPC yield for SHBT-R.

This module models:
  * 72 GHz Dynamical Casimir Effect (DCE) pair-generation producing 36 GHz
    two-mode squeezed vacuum with squeezing parameter r = 1.52.
  * Electro-optic conversion of the 36 GHz microwave squeezing to the
    193.41 THz (1550 nm) optical domain, including power loss eta and
    phase jitter sigma_theta <= 0.01 rad.
  * Statistical Process Control (SPC) Monte-Carlo yield analysis over
    52 hexamer clusters (312 channels) with CD 250.12 +/- 0.42 nm and
    sidewall angle 89.88 +/- 0.04 deg.
  * A 312-channel hexamer photonic microcavity network.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class ProcessParameters:
    """Nanofabrication process variation parameters."""

    n_aerogel_nom: float = 1.0182
    n_aerogel_sigma: float = 0.0003
    sidewall_nom_deg: float = 89.88
    sidewall_sigma_deg: float = 0.04
    cd_width_nom_nm: float = 250.12
    cd_width_sigma_nm: float = 0.42
    temperature_k: float = 4.2


@dataclass
class ResonanceSensitivities:
    """Sensitivity of ring resonance wavelength to process parameters."""

    lambda_center_nm: float = 1550.0
    ng: float = 4.2
    tol_nm: float = 0.05
    dneff_dn: float = 0.055
    dneff_dw: float = 1.1e-4       # per nm of CD
    dneff_dtheta: float = 9.0e-4   # per degree sidewall


@dataclass
class SPCResult:
    """Output of a Monte-Carlo SPC yield run."""

    n_wafers: int
    n_dies: int
    total_channels: int
    sigma_lambda_analytic_nm: float
    yield_analytic: float
    total_yield: float
    out_of_control_wafers: np.ndarray
    per_wafer_mean_nm: np.ndarray
    per_wafer_std_nm: np.ndarray
    cd_width_mean_nm: float
    cd_width_std_nm: float
    sidewall_mean_deg: float
    sidewall_std_deg: float
    n_aerogel_mean: float
    n_aerogel_std: float
    dce_r: float
    eo_squeezed_var: float
    network_expected_yield: float


class MonteCarloSPCYieldEngine:
    """Monte-Carlo SPC yield engine for the SHBT-R photonic PDK."""

    def __init__(
        self,
        process: Optional[ProcessParameters] = None,
        sens: Optional[ResonanceSensitivities] = None,
        n_wafers: int = 52,
        dies_per_wafer: int = 6,
        rng: Optional[np.random.Generator] = None,
    ):
        self.process = process or ProcessParameters()
        self.sens = sens or ResonanceSensitivities()
        self.n_wafers = int(n_wafers)
        self.dies_per_wafer = int(dies_per_wafer)
        self.rng = rng or np.random.default_rng(0)

    def _delta_lambda_analytic_nm(self) -> float:
        p = self.process
        s = self.sens
        sigma_neff = math.sqrt(
            (s.dneff_dn * p.n_aerogel_sigma) ** 2
            + (s.dneff_dw * p.cd_width_sigma_nm) ** 2
            + (s.dneff_dtheta * p.sidewall_sigma_deg) ** 2
        )
        return (s.lambda_center_nm / s.ng) * sigma_neff

    def _analytic_yield(self) -> float:
        sigma = self._delta_lambda_analytic_nm()
        if sigma <= 0.0:
            return 1.0
        z = self.sens.tol_nm / (sigma * math.sqrt(2))
        return math.erf(z)

    @staticmethod
    def _c4(n: int) -> float:
        """Control-chart constant for the unbiased standard deviation."""
        if n <= 1:
            return 1.0
        return math.sqrt(2.0 / (n - 1.0)) * math.gamma(n / 2.0) / math.gamma((n - 1.0) / 2.0)

    def run(self) -> SPCResult:
        p = self.process
        s = self.sens
        n = self.dies_per_wafer

        shape = (self.n_wafers, n)
        n_aero = self.rng.normal(p.n_aerogel_nom, p.n_aerogel_sigma, size=shape)
        cd = self.rng.normal(p.cd_width_nom_nm, p.cd_width_sigma_nm, size=shape)
        sw = self.rng.normal(p.sidewall_nom_deg, p.sidewall_sigma_deg, size=shape)

        delta_neff = (
            s.dneff_dn * (n_aero - p.n_aerogel_nom)
            + s.dneff_dw * (cd - p.cd_width_nom_nm)
            + s.dneff_dtheta * (sw - p.sidewall_nom_deg)
        )
        delta_lambda_nm = (s.lambda_center_nm / s.ng) * delta_neff

        within_tol = np.abs(delta_lambda_nm) <= s.tol_nm
        total_yield = float(np.mean(within_tol))

        per_wafer_mean_nm = np.mean(delta_lambda_nm, axis=1)
        per_wafer_std_nm = np.std(delta_lambda_nm, axis=1, ddof=1)

        sigma_analytic = self._delta_lambda_analytic_nm()
        yield_analytic = self._analytic_yield()

        c4 = self._c4(n)
        # X-bar and S-chart control limits using the analytic sigma as the
        # known process standard deviation.
        a3 = 3.0 / (c4 * math.sqrt(n)) if n > 1 else 0.0
        xbar_ucl = a3 * sigma_analytic
        xbar_lcl = -xbar_ucl

        b4 = 1.0 + 3.0 / c4 * math.sqrt(1.0 - c4 * c4) if n > 1 else 1.0
        b3 = max(0.0, 1.0 - 3.0 / c4 * math.sqrt(1.0 - c4 * c4)) if n > 1 else 0.0
        s_ucl = b4 * sigma_analytic
        s_lcl = b3 * sigma_analytic

        ooc = np.where(
            (per_wafer_mean_nm > xbar_ucl)
            | (per_wafer_mean_nm < xbar_lcl)
            | (per_wafer_std_nm > s_ucl)
            | (per_wafer_std_nm < s_lcl)
        )[0]

        return SPCResult(
            n_wafers=self.n_wafers,
            n_dies=self.n_wafers * n,
            total_channels=self.n_wafers * n,
            sigma_lambda_analytic_nm=sigma_analytic,
            yield_analytic=yield_analytic,
            total_yield=total_yield,
            out_of_control_wafers=ooc,
            per_wafer_mean_nm=per_wafer_mean_nm,
            per_wafer_std_nm=per_wafer_std_nm,
            cd_width_mean_nm=float(np.mean(cd)),
            cd_width_std_nm=float(np.std(cd, ddof=1)),
            sidewall_mean_deg=float(np.mean(sw)),
            sidewall_std_deg=float(np.std(sw, ddof=1)),
            n_aerogel_mean=float(np.mean(n_aero)),
            n_aerogel_std=float(np.std(n_aero, ddof=1)),
            dce_r=1.52,
            eo_squeezed_var=0.0,
            network_expected_yield=0.0,
        )


class PhotonicMicrocavityNetwork:
    """52-cluster hexamer microcavity network with 312 channels."""

    def __init__(
        self,
        num_channels: int = 312,
        clusters: int = 52,
        channels_per_cluster: Optional[int] = None,
        wavelength_nm: float = 1550.0,
        cluster_radius_um: float = 12.0,
        rng: Optional[np.random.Generator] = None,
    ):
        if channels_per_cluster is None:
            if num_channels % clusters != 0:
                raise ValueError("num_channels must be divisible by clusters")
            channels_per_cluster = num_channels // clusters
        self.num_channels = int(num_channels)
        self.clusters = int(clusters)
        self.channels_per_cluster = int(channels_per_cluster)
        self.wavelength_nm = float(wavelength_nm)
        self.cluster_radius_um = float(cluster_radius_um)
        self.rng = rng or np.random.default_rng(0)

    def cluster_assignments(self) -> np.ndarray:
        return np.repeat(np.arange(self.clusters), self.channels_per_cluster)

    def expected_yield(self, sigma_lambda_nm: float = 0.0225) -> float:
        """Expected spectral yield for a Gaussian wavelength tolerance."""
        if sigma_lambda_nm <= 0.0:
            return 1.0
        z = 0.05 / (sigma_lambda_nm * math.sqrt(2))
        return math.erf(z)

    def channel_wavelengths(self) -> np.ndarray:
        """Return nominal center wavelength for each channel."""
        return np.full(self.num_channels, self.wavelength_nm)


class DynamicalCasimirSqueezer:
    """72 GHz DCE pump producing 36 GHz two-mode squeezed vacuum."""

    HBAR = 1.054571817e-34
    HPLANCK = 2.0 * math.pi * HBAR

    def __init__(
        self,
        pump_freq: float = 72.0e9,
        squeeze_freq: float = 36.0e9,
        r: float = 1.52,
        chi_dce: Optional[float] = None,
    ):
        self.pump_freq = float(pump_freq)
        self.squeeze_freq = float(squeeze_freq)
        self.r = float(r)
        self.chi_dce = chi_dce

    def _squeezing_time(self) -> float:
        """Effective interaction time implied by r = chi * t."""
        if self.chi_dce and self.chi_dce > 0.0:
            return self.r / self.chi_dce
        # Default: 1 ns interaction.
        return 1.0e-9

    def two_mode_squeezed_covariance(self) -> np.ndarray:
        """Return the 4x4 covariance matrix (X1,P1,X2,P2) in hbar/2 units."""
        c = math.cosh(2.0 * self.r)
        s = math.sinh(2.0 * self.r)
        # Vacuum covariance is 0.5 * I4; the squeezed covariance uses the same
        # convention so that the symplectic eigenvalues are {0.5, 0.5}.
        V = 0.5 * np.array(
            [
                [c, 0.0, s, 0.0],
                [0.0, c, 0.0, -s],
                [s, 0.0, c, 0.0],
                [0.0, -s, 0.0, c],
            ],
            dtype=float,
        )
        return V

    @staticmethod
    def symplectic_eigenvalues(cov: np.ndarray) -> np.ndarray:
        """Compute the symplectic eigenvalues of a 2-mode covariance matrix."""
        V = np.asarray(cov, dtype=float)
        A = V[:2, :2]
        B = V[2:, 2:]
        C = V[:2, 2:]
        detA = float(np.linalg.det(A))
        detB = float(np.linalg.det(B))
        detC = float(np.linalg.det(C))
        detV = float(np.linalg.det(V))
        Delta = detA + detB + 2.0 * detC
        disc = max(0.0, Delta * Delta - 4.0 * detV)
        nu_sq_plus = (Delta + math.sqrt(disc)) / 2.0
        nu_sq_minus = (Delta - math.sqrt(disc)) / 2.0
        return np.array([math.sqrt(max(0.0, nu_sq_plus)), math.sqrt(max(0.0, nu_sq_minus))])

    @staticmethod
    def combined_quadrature_variance(cov: np.ndarray) -> float:
        """Variance of the squeezed (X1 - X2)/sqrt(2) EPR quadrature."""
        V = np.asarray(cov, dtype=float)
        return 0.5 * (V[0, 0] + V[2, 2] - 2.0 * V[0, 2])

    def squeezed_quadrature_variance(self) -> float:
        """Shortcut for the squeezed EPR quadrature variance of this squeezer."""
        return self.combined_quadrature_variance(self.two_mode_squeezed_covariance())


class ElectroOpticConverter:
    """Electro-optic conversion from 36 GHz microwave squeezing to 1550 nm."""

    C0 = 299_792_458.0

    def __init__(
        self,
        optical_wavelength: float = 1550.0e-9,
        eta: float = 0.95,
        sigma_theta: float = 0.01,
    ):
        self.optical_wavelength = float(optical_wavelength)
        self.eta = float(eta)
        self.sigma_theta = float(sigma_theta)
        self.optical_freq = self.C0 / self.optical_wavelength

    def _rotation_matrix(self, theta: float) -> np.ndarray:
        ct = math.cos(theta)
        st = math.sin(theta)
        return np.array([[ct, -st], [st, ct]], dtype=float)

    def _conversion_matrix(self, theta: float) -> np.ndarray:
        """4x4 block-diagonal matrix converting both microwave modes to optical."""
        sqrt_eta = math.sqrt(self.eta)
        R = self._rotation_matrix(theta)
        M = np.zeros((4, 4), dtype=float)
        M[:2, :2] = sqrt_eta * R
        M[2:, 2:] = sqrt_eta * R
        return M

    def convert(
        self,
        cov: np.ndarray,
        theta: float = 0.0,
        apply_jitter: bool = False,
        n_samples: int = 200,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Convert a two-mode microwave squeezed covariance to optical.

        The output is the microwave quadrature covariance transformed by the
        EO matrix with optical loss and (optionally) averaged over phase jitter.
        """
        if rng is None:
            rng = np.random.default_rng(0)
        cov = np.asarray(cov, dtype=float)
        noise = (1.0 - self.eta) * 0.5 * np.eye(4)

        if not apply_jitter:
            M = self._conversion_matrix(theta)
            return M @ cov @ M.T + noise

        thetas = rng.normal(0.0, self.sigma_theta, size=n_samples)
        V_out = np.zeros_like(cov)
        for t in thetas:
            M = self._conversion_matrix(t)
            V_out += M @ cov @ M.T
        V_out = V_out / n_samples + noise
        return V_out


# ---------------------------------------------------------------------------
# Unified self-test entry point
# ---------------------------------------------------------------------------


def run_cad_engine_tests() -> None:
    """Run all photonic CAD, DCE, EO and SPC self-tests."""
    print("Running SHBT-R photonic CAD / DCE / EO / SPC self-tests...")
    rng = np.random.default_rng(42)

    # --- SPC Monte Carlo over 52 hexamer clusters (312 channels) ---
    process = ProcessParameters()
    sens = ResonanceSensitivities()
    spc = MonteCarloSPCYieldEngine(
        process=process,
        sens=sens,
        n_wafers=52,
        dies_per_wafer=6,
        rng=rng,
    )
    result = spc.run()

    assert result.n_wafers == 52
    assert result.n_dies == 312
    assert result.total_channels == 312
    assert 0.0 <= result.total_yield <= 1.0
    assert result.total_yield > 0.80, "SPC yield too low"
    assert abs(result.total_yield - result.yield_analytic) < 0.05, "MC / analytic yield mismatch"

    assert abs(result.cd_width_mean_nm - 250.12) < 0.05
    assert abs(result.cd_width_std_nm - 0.42) < 0.3 * 0.42
    assert abs(result.sidewall_mean_deg - 89.88) < 0.05
    assert abs(result.sidewall_std_deg - 0.04) < 0.3 * 0.04
    assert abs(result.n_aerogel_mean - 1.0182) < 0.0002
    assert abs(result.n_aerogel_std - 0.0003) < 0.3 * 0.0003

    # --- DCE two-mode squeezer ---
    dce = DynamicalCasimirSqueezer(
        pump_freq=72.0e9,
        squeeze_freq=36.0e9,
        r=1.52,
    )
    cov_mw = dce.two_mode_squeezed_covariance()
    assert cov_mw.shape == (4, 4)
    assert np.allclose(cov_mw, cov_mw.T, atol=1e-12)
    symp = dce.symplectic_eigenvalues(cov_mw)
    assert np.all((symp > 0.48) & (symp < 0.52)), "symplectic eigenvalues not at vacuum"
    assert dce.r == 1.52

    # --- Electro-optic conversion to 1550 nm ---
    eo = ElectroOpticConverter(
        optical_wavelength=1550.0e-9,
        eta=0.95,
        sigma_theta=0.01,
    )
    cov_opt = eo.convert(cov_mw, apply_jitter=False)
    assert cov_opt.shape == (4, 4)
    assert np.allclose(cov_opt, cov_opt.T, atol=1e-12)

    squeezed_in = dce.combined_quadrature_variance(cov_mw)
    squeezed_out = dce.combined_quadrature_variance(cov_opt)
    assert squeezed_out < 0.5, "squeezing lost in EO conversion"
    # Loss degrades squeezing to eta * v + (1-eta)/2; allow small margin.
    assert squeezed_out < squeezed_in + 0.06

    # --- Hexamer microcavity network ---
    net = PhotonicMicrocavityNetwork(num_channels=312, clusters=52, rng=rng)
    assert net.channels_per_cluster == 6
    assert 0.0 < net.expected_yield(sigma_lambda_nm=result.sigma_lambda_analytic_nm) < 1.0
    assert len(net.cluster_assignments()) == 312

    # Fill in the extra result fields for any downstream reporting.
    result.dce_r = dce.r
    result.eo_squeezed_var = squeezed_out
    result.network_expected_yield = net.expected_yield(
        sigma_lambda_nm=result.sigma_lambda_analytic_nm
    )

    print(f"  SPC yield (analytic / MC): {result.yield_analytic:.3f} / {result.total_yield:.3f}")
    print(f"  DCE squeezing parameter r = {dce.r:.3f}")
    print(f"  EO squeezed quadrature variance = {squeezed_out:.4f}")
    print(f"  Hexamer network expected yield = {result.network_expected_yield:.3f}")
    print("All SHBT-R photonic CAD self-tests passed.")


if __name__ == "__main__":
    run_cad_engine_tests()
