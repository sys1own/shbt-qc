"""Photonic fabrication-tolerance and network-synthesis toolchain for SHBT-R.

Two engines are provided:

1. MonteCarloSPCYieldEngine - Monte Carlo statistical process control across a
   100-wafer x 100-die dataset.  Aerogel cladding index, sidewall angle, and
   critical-dimension (CD) width are sampled as Gaussian process variables with
   wafer-level and die-level components.  Each die's microring resonance shift
   is evaluated against a +/- 0.05 nm tolerance window and the wafer/lot yield,
   process capability (Cp/Cpk) and X-bar/S control-chart limits are produced.

2. PhotonicMicrocavityNetwork - 312-channel microcavity network organised into
   52 hexamer (6-ring) clusters.  Constructs the full inter-cavity evanescent
   coupling matrix (nearest-neighbour ring coupling J_ring = 25.4 GHz, plus a
   weaker inter-cluster bus coupling), diagonalises it for the supermode
   spectrum, and evaluates the third-order Kerr nonlinear phase shift phi_kerr
   for every channel.

``run_cad_engine_tests()`` verifies non-trivial execution and asserts that the
SPC yield exceeds 80 % under standard Gaussian process parameters.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# Physical constants
C0 = 299_792_458.0   # m/s
HBAR = 1.054571817e-34
H_PLANCK = 2.0 * math.pi * HBAR


@dataclass
class ProcessParameters:
    """Nominal values and 1-sigma spreads of the photonic process variables."""

    n_aerogel_nom: float = 1.0182
    n_aerogel_sigma: float = 0.0003
    sidewall_nom_deg: float = 89.88
    sidewall_sigma_deg: float = 0.05
    cd_width_nom_nm: float = 480.0
    cd_width_sigma_nm: float = 0.6
    # Fraction of each variance that is shared by all dies on a wafer
    wafer_fraction: float = 0.1


@dataclass
class ResonanceSensitivities:
    """First-order resonance-wavelength sensitivities of the microring design.

    Values are derived for a 480 nm x 220 nm Si-on-aerogel strip waveguide near
    1550 nm.  The aerogel is the low-index cladding, so the effective index has a
    weak dependence on n_aerogel (dn_eff/dn_clad ~ 0.055) compared with a
    conventional oxide cladding.
    """

    wavelength_nm: float = 1550.0
    n_group: float = 4.2
    dneff_dn_clad: float = 0.055          # RIU / RIU
    dneff_dw: float = 1.1e-4              # RIU / nm  (CD width)
    dneff_dtheta: float = 9.0e-4          # RIU / deg (sidewall angle)

    def dlambda_dneff(self) -> float:
        """d(lambda_res)/d(n_eff) = lambda / n_g   [nm / RIU]."""
        return self.wavelength_nm / self.n_group


@dataclass
class SPCResult:
    """Container for a Monte Carlo SPC run."""

    resonance_shift_nm: np.ndarray        # (wafers, dies)
    die_pass: np.ndarray                  # (wafers, dies) bool
    wafer_yield: np.ndarray               # (wafers,)
    total_yield: float
    cp: float
    cpk: float
    xbar_ucl: float
    xbar_lcl: float
    s_ucl: float
    s_lcl: float
    out_of_control_wafers: np.ndarray     # indices
    stats: Dict[str, float] = field(default_factory=dict)


class MonteCarloSPCYieldEngine:
    """Monte Carlo statistical process control for microring resonance yield."""

    # Unbiased estimator constants for X-bar/S charts (n = 100 subgroup size)
    _C4_N100 = 0.99748

    def __init__(
        self,
        n_wafers: int = 100,
        dies_per_wafer: int = 100,
        tolerance_nm: float = 0.05,
        process: Optional[ProcessParameters] = None,
        sensitivities: Optional[ResonanceSensitivities] = None,
        seed: Optional[int] = 20260908,
    ) -> None:
        if n_wafers <= 0 or dies_per_wafer <= 0:
            raise ValueError("n_wafers and dies_per_wafer must be positive")
        self.n_wafers = int(n_wafers)
        self.dies_per_wafer = int(dies_per_wafer)
        self.tolerance_nm = float(tolerance_nm)
        self.process = process or ProcessParameters()
        self.sens = sensitivities or ResonanceSensitivities()
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def _sample_hierarchical(self, nominal: float, sigma: float) -> np.ndarray:
        """Gaussian samples with shared wafer-level and independent die-level parts."""
        shape = (self.n_wafers, self.dies_per_wafer)
        wf = self.process.wafer_fraction
        sigma_wafer = sigma * math.sqrt(wf)
        sigma_die = sigma * math.sqrt(1.0 - wf)
        wafer_offset = self.rng.normal(0.0, sigma_wafer, size=(self.n_wafers, 1))
        die_noise = self.rng.normal(0.0, sigma_die, size=shape)
        return nominal + wafer_offset + die_noise

    def sample_process(self) -> Dict[str, np.ndarray]:
        """Draw the full (wafers x dies) process-variable dataset."""
        p = self.process
        return {
            "n_aerogel": self._sample_hierarchical(p.n_aerogel_nom, p.n_aerogel_sigma),
            "sidewall_deg": self._sample_hierarchical(p.sidewall_nom_deg, p.sidewall_sigma_deg),
            "cd_width_nm": self._sample_hierarchical(p.cd_width_nom_nm, p.cd_width_sigma_nm),
        }

    # ------------------------------------------------------------------
    # Physics
    # ------------------------------------------------------------------
    def effective_index_shift(self, samples: Dict[str, np.ndarray]) -> np.ndarray:
        """First-order dn_eff of every die relative to the nominal design."""
        p, s = self.process, self.sens
        dn_clad = samples["n_aerogel"] - p.n_aerogel_nom
        dw = samples["cd_width_nm"] - p.cd_width_nom_nm
        dtheta = samples["sidewall_deg"] - p.sidewall_nom_deg
        return s.dneff_dn_clad * dn_clad + s.dneff_dw * dw + s.dneff_dtheta * dtheta

    def resonance_shift_nm(self, samples: Dict[str, np.ndarray]) -> np.ndarray:
        """Resonance-wavelength shift of every die [nm]."""
        return self.sens.dlambda_dneff() * self.effective_index_shift(samples)

    def analytic_sigma_nm(self) -> float:
        """Closed-form 1-sigma resonance shift from linear error propagation."""
        p, s = self.process, self.sens
        var = (
            (s.dneff_dn_clad * p.n_aerogel_sigma) ** 2
            + (s.dneff_dw * p.cd_width_sigma_nm) ** 2
            + (s.dneff_dtheta * p.sidewall_sigma_deg) ** 2
        )
        return s.dlambda_dneff() * math.sqrt(var)

    def analytic_yield(self) -> float:
        """Expected yield for a centred Gaussian within +/- tolerance."""
        z = self.tolerance_nm / self.analytic_sigma_nm()
        return math.erf(z / math.sqrt(2.0))

    # ------------------------------------------------------------------
    # SPC
    # ------------------------------------------------------------------
    def run(self) -> SPCResult:
        samples = self.sample_process()
        shift = self.resonance_shift_nm(samples)
        die_pass = np.abs(shift) <= self.tolerance_nm
        wafer_yield = die_pass.mean(axis=1)
        total_yield = float(die_pass.mean())

        mu = float(shift.mean())
        sigma = float(shift.std(ddof=1))
        usl, lsl = self.tolerance_nm, -self.tolerance_nm
        cp = (usl - lsl) / (6.0 * sigma)
        cpk = min(usl - mu, mu - lsl) / (3.0 * sigma)

        # X-bar / S control chart over wafers (subgroup = wafer)
        xbar = shift.mean(axis=1)
        s_w = shift.std(axis=1, ddof=1)
        s_bar = float(s_w.mean())
        n = self.dies_per_wafer
        c4 = self._C4_N100 if n == 100 else math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)
        a3 = 3.0 / (c4 * math.sqrt(n))
        b3 = max(0.0, 1.0 - 3.0 / c4 * math.sqrt(1.0 - c4 ** 2))
        b4 = 1.0 + 3.0 / c4 * math.sqrt(1.0 - c4 ** 2)
        xbar_ucl = mu + a3 * s_bar
        xbar_lcl = mu - a3 * s_bar
        s_ucl = b4 * s_bar
        s_lcl = b3 * s_bar
        ooc = np.where((xbar > xbar_ucl) | (xbar < xbar_lcl) | (s_w > s_ucl) | (s_w < s_lcl))[0]

        stats = {
            "mean_shift_nm": mu,
            "sigma_shift_nm": sigma,
            "analytic_sigma_nm": self.analytic_sigma_nm(),
            "analytic_yield": self.analytic_yield(),
            "min_wafer_yield": float(wafer_yield.min()),
            "max_wafer_yield": float(wafer_yield.max()),
            "n_aerogel_mean": float(samples["n_aerogel"].mean()),
            "n_aerogel_std": float(samples["n_aerogel"].std(ddof=1)),
            "sidewall_mean_deg": float(samples["sidewall_deg"].mean()),
            "sidewall_std_deg": float(samples["sidewall_deg"].std(ddof=1)),
            "cd_width_mean_nm": float(samples["cd_width_nm"].mean()),
            "cd_width_std_nm": float(samples["cd_width_nm"].std(ddof=1)),
        }
        return SPCResult(
            resonance_shift_nm=shift,
            die_pass=die_pass,
            wafer_yield=wafer_yield,
            total_yield=total_yield,
            cp=float(cp),
            cpk=float(cpk),
            xbar_ucl=float(xbar_ucl),
            xbar_lcl=float(xbar_lcl),
            s_ucl=float(s_ucl),
            s_lcl=float(s_lcl),
            out_of_control_wafers=ooc,
            stats=stats,
        )


class PhotonicMicrocavityNetwork:
    """312-channel microcavity network of 52 hexamer ring clusters.

    Coupling model (angular frequency units, Hz x 2pi handled internally):

    * Intra-hexamer: each ring couples evanescently to its two ring neighbours
      with strength J_ring (closed 6-ring loop).
    * Inter-hexamer: ring 0 of cluster k couples to ring 3 of cluster k+1 through
      a bus waveguide with strength J_bus (open chain of clusters).
    * Diagonal: cavity resonance detuning delta_k relative to the network
      centre frequency.

    Kerr phase per channel over one round trip:

        phi_kerr = (2*pi / lambda) * n2 * I * L_rt,   I = P_circ / A_eff
        P_circ = P_in * (F / pi)  (build-up in a critically coupled ring)
    """

    def __init__(
        self,
        n_channels: int = 312,
        cluster_size: int = 6,
        J_ring_hz: float = 25.4e9,
        J_bus_hz: float = 2.5e9,
        detuning_sigma_hz: float = 1.0e9,
        wavelength_m: float = 1.55e-6,
        ring_radius_m: float = 5.0e-6,
        n2_m2_per_w: float = 4.5e-18,
        a_eff_m2: float = 0.1e-12,
        finesse: float = 60.0,
        seed: Optional[int] = 7,
    ) -> None:
        if n_channels % cluster_size != 0:
            raise ValueError("n_channels must be a multiple of cluster_size")
        self.N = int(n_channels)
        self.cluster_size = int(cluster_size)
        self.n_clusters = self.N // self.cluster_size
        self.J_ring = float(J_ring_hz)
        self.J_bus = float(J_bus_hz)
        self.detuning_sigma = float(detuning_sigma_hz)
        self.wavelength = float(wavelength_m)
        self.ring_radius = float(ring_radius_m)
        self.n2 = float(n2_m2_per_w)
        self.a_eff = float(a_eff_m2)
        self.finesse = float(finesse)
        self.rng = np.random.default_rng(seed)
        self.detuning_hz = self.rng.normal(0.0, self.detuning_sigma, size=self.N)

    # ------------------------------------------------------------------
    # Topology
    # ------------------------------------------------------------------
    def cluster_of(self, channel: int) -> int:
        return channel // self.cluster_size

    def coupling_matrix(self) -> np.ndarray:
        """Full N x N Hermitian evanescent coupling matrix in Hz."""
        H = np.zeros((self.N, self.N), dtype=float)
        m = self.cluster_size
        for c in range(self.n_clusters):
            base = c * m
            for i in range(m):
                a = base + i
                b = base + (i + 1) % m
                H[a, b] = H[b, a] = self.J_ring
            if c + 1 < self.n_clusters:
                a = base            # ring 0 of cluster c
                b = base + m + m // 2  # ring 3 of cluster c+1
                H[a, b] = H[b, a] = self.J_bus
        H[np.diag_indices(self.N)] = self.detuning_hz
        return H

    def supermode_spectrum(self) -> tuple:
        """Eigenfrequencies (Hz) and eigenvectors of the coupling matrix."""
        H = self.coupling_matrix()
        return np.linalg.eigh(H)

    def hexamer_ideal_spectrum(self) -> np.ndarray:
        """Analytic supermodes of a closed 6-ring loop: 2 J cos(2 pi k / 6)."""
        k = np.arange(self.cluster_size)
        return 2.0 * self.J_ring * np.cos(2.0 * math.pi * k / self.cluster_size)

    # ------------------------------------------------------------------
    # Kerr nonlinearity
    # ------------------------------------------------------------------
    def round_trip_length(self) -> float:
        return 2.0 * math.pi * self.ring_radius

    def circulating_power(self, p_in_w: np.ndarray) -> np.ndarray:
        return np.asarray(p_in_w, dtype=float) * self.finesse / math.pi

    def kerr_phase_shift(self, p_in_w) -> np.ndarray:
        """Third-order Kerr phase phi_kerr [rad] for every channel."""
        p_in = np.broadcast_to(np.asarray(p_in_w, dtype=float), (self.N,))
        intensity = self.circulating_power(p_in) / self.a_eff
        return (2.0 * math.pi / self.wavelength) * self.n2 * intensity * self.round_trip_length()

    def kerr_frequency_shift_hz(self, p_in_w) -> np.ndarray:
        """Resonance shift induced by phi_kerr: df = -phi_kerr * FSR / (2 pi)."""
        n_g = 4.2
        fsr = C0 / (n_g * self.round_trip_length())
        return -self.kerr_phase_shift(p_in_w) * fsr / (2.0 * math.pi)

    def kerr_shifted_coupling_matrix(self, p_in_w) -> np.ndarray:
        """Coupling matrix with Kerr-shifted diagonal detunings."""
        H = self.coupling_matrix()
        H[np.diag_indices(self.N)] += self.kerr_frequency_shift_hz(p_in_w)
        return H


def run_cad_engine_tests() -> None:
    """Self-contained execution verification for the CAD/yield toolchain."""
    print("Running SHBT-R photonic CAD / SPC self-tests...")

    # ------------------------------------------------------------------
    # 1. Monte Carlo SPC yield
    # ------------------------------------------------------------------
    engine = MonteCarloSPCYieldEngine(n_wafers=100, dies_per_wafer=100, tolerance_nm=0.05)
    result = engine.run()

    assert result.resonance_shift_nm.shape == (100, 100), "Expected 100 wafers x 100 dies"
    assert result.die_pass.shape == (100, 100)
    assert np.all(np.isfinite(result.resonance_shift_nm)), "Non-finite resonance shifts"
    assert result.resonance_shift_nm.std() > 0.0, "Process variation must be non-trivial"

    st = result.stats
    assert abs(st["n_aerogel_mean"] - 1.0182) < 5e-5, "Aerogel index mean drifted"
    assert abs(st["n_aerogel_std"] - 0.0003) < 0.3 * 0.0003, "Aerogel index sigma mismatch"
    assert abs(st["sidewall_mean_deg"] - 89.88) < 0.01, "Sidewall mean drifted"
    assert abs(st["sidewall_std_deg"] - 0.05) < 0.3 * 0.05, "Sidewall sigma mismatch"
    assert abs(st["sigma_shift_nm"] - st["analytic_sigma_nm"]) < 0.25 * st["analytic_sigma_nm"], \
        "Monte Carlo sigma disagrees with linear error propagation"

    assert result.total_yield > 0.80, f"SPC yield {result.total_yield:.3f} must exceed 80%"
    assert result.total_yield < 1.0, "Yield of exactly 100% indicates a trivial tolerance test"
    assert abs(result.total_yield - st["analytic_yield"]) < 0.05, "MC yield disagrees with analytic yield"
    assert result.cpk > 0.0 and result.cpk <= result.cp + 1e-12, "Cpk must be positive and <= Cp"
    assert result.s_ucl > result.s_lcl >= 0.0
    assert result.xbar_ucl > result.xbar_lcl

    print(f"  SPC: yield = {100*result.total_yield:.2f}%  (analytic {100*st['analytic_yield']:.2f}%), "
          f"sigma = {st['sigma_shift_nm']*1e3:.2f} pm, Cp = {result.cp:.2f}, Cpk = {result.cpk:.2f}, "
          f"OOC wafers = {result.out_of_control_wafers.size}")

    # Degraded process must lower yield (sanity on the tolerance test)
    bad = ProcessParameters(n_aerogel_sigma=0.0015, sidewall_sigma_deg=0.25, cd_width_sigma_nm=3.0)
    degraded = MonteCarloSPCYieldEngine(process=bad, seed=1).run()
    assert degraded.total_yield < result.total_yield, "Degraded process should lower yield"

    # ------------------------------------------------------------------
    # 2. Photonic microcavity network
    # ------------------------------------------------------------------
    net = PhotonicMicrocavityNetwork(n_channels=312, cluster_size=6, J_ring_hz=25.4e9)
    assert net.n_clusters == 52, "312 channels must form 52 hexamers"

    H = net.coupling_matrix()
    assert H.shape == (312, 312)
    assert np.allclose(H, H.T), "Coupling matrix must be symmetric (Hermitian)"
    off = H - np.diag(np.diag(H))
    n_ring_links = int(np.sum(np.isclose(off, 25.4e9))) // 2
    n_bus_links = int(np.sum(np.isclose(off, net.J_bus))) // 2
    assert n_ring_links == 52 * 6, f"Expected 312 intra-hexamer couplings, got {n_ring_links}"
    assert n_bus_links == 51, f"Expected 51 inter-cluster bus couplings, got {n_bus_links}"
    # Each ring has exactly two ring neighbours
    ring_degree = np.sum(np.isclose(off, 25.4e9), axis=1)
    assert np.all(ring_degree == 2), "Every ring must have two intra-hexamer neighbours"

    evals, evecs = net.supermode_spectrum()
    assert evals.size == 312 and np.all(np.isfinite(evals))
    ideal = net.hexamer_ideal_spectrum()
    # Supermode band must span the ideal hexamer band (+/- 2J) up to detuning/bus perturbations
    assert evals.max() > 0.9 * ideal.max() and evals.min() < 0.9 * ideal.min(), \
        "Supermode spectrum should span the hexamer +/- 2J band"
    assert np.allclose(evecs @ np.diag(evals) @ evecs.T, H, atol=1e-3 * np.abs(H).max()), \
        "Eigen-decomposition must reconstruct H"

    # Isolated hexamer check (zero detuning, no bus): eigenvalues == 2J cos(2 pi k/6)
    iso = PhotonicMicrocavityNetwork(n_channels=6, cluster_size=6, J_ring_hz=25.4e9,
                                     J_bus_hz=0.0, detuning_sigma_hz=0.0)
    iso_evals, _ = iso.supermode_spectrum()
    assert np.allclose(np.sort(iso_evals), np.sort(iso.hexamer_ideal_spectrum()), rtol=1e-9, atol=1.0), \
        "Isolated hexamer spectrum must match analytic 2J cos(2 pi k/6)"

    p_in = np.full(312, 1.0e-3)  # 1 mW per channel
    phi = net.kerr_phase_shift(p_in)
    assert phi.shape == (312,)
    assert np.all(phi > 0.0) and np.all(np.isfinite(phi)), "Kerr phase must be positive and finite"
    phi_2mw = net.kerr_phase_shift(2.0e-3)
    assert np.allclose(phi_2mw, 2.0 * phi), "Kerr phase must scale linearly with input power"
    H_kerr = net.kerr_shifted_coupling_matrix(p_in)
    assert not np.allclose(np.diag(H_kerr), np.diag(H)), "Kerr shift must perturb resonances"
    assert np.allclose(H_kerr - np.diag(np.diag(H_kerr)), off), "Kerr must not alter couplings"

    print(f"  Network: 312 ch / {net.n_clusters} hexamers, supermode band "
          f"[{evals.min()/1e9:.2f}, {evals.max()/1e9:.2f}] GHz, "
          f"phi_kerr(1 mW) = {phi.mean()*1e3:.3f} mrad, "
          f"Kerr shift = {net.kerr_frequency_shift_hz(p_in).mean()/1e6:.2f} MHz")

    print("All photonic CAD / SPC self-tests passed.")


if __name__ == "__main__":
    run_cad_engine_tests()
