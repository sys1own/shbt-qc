"""Multi-physics simulation core for the SHBT-R Quantum Computer.

This module provides self-contained Python solvers for the dominant engineering
domains of a cryogenic photonic/ electronic quantum-computing substrate:

1. CryogenicThermalSolver   - 3-D finite-difference thermal diffusion across
   28 nm FD-SOI and InP at 4.2 K with T^3 specific heat and thermal
   conductivity and Kapitza interface resistance.

2. VibrationJitterFEA       - 1 Hz - 500 Hz vibration spectrum, piezo
   feed-forward cancellation, and resulting optical waveguide phase jitter.

3. RogersInterposerSIPISolver - 12-layer Rogers RO4350B interposer signal and
   power-integrity metrics: S21 insertion loss, Far-End Crosstalk (FEXT), and
   PDN impedance Z_PDN(f).

A built-in test harness ``run_multiphysics_tests()`` validates non-trivial
execution of every solver.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np

# Physical constants
C0 = 299_792_458.0          # m/s
MU0 = 4.0 * math.pi * 1e-7  # H/m
EPS0 = 8.854187817e-12      # F/m


def _to_array(value: Union[float, np.ndarray], shape: tuple) -> np.ndarray:
    """Broadcast a scalar or array to ``shape`` as a float ndarray."""
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return np.full(shape, float(value), dtype=float)
    if arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


class CryogenicThermalSolver:
    """3-D finite-difference cryogenic thermal diffusion solver.

    The model treats the substrate as a rectangular grid with two materials:

    * FD-SOI (silicon) in the lower ``fd_soi_thickness`` region
    * InP in the remaining upper region

    Both materials use low-temperature Debye-like laws:

        C_p(T) = cp_factor * T^3   [J kg^-1 K^-1]
        k(T)   = k_factor   * T^3   [W m^-1 K^-1]

    Thermal coupling between neighbouring control volumes uses a finite-volume
    conductance that includes a Kapitza resistance ``R_K`` whenever the two
    volumes are of a different material or when a volume faces the 4.2 K bath.
    """

    def __init__(
        self,
        shape: Sequence[int] = (18, 18, 18),
        spacing: float = 5.0e-9,
        T_bath: float = 4.2,
        fd_soi_thickness: float = 28.0e-9,
        fbd_cp_factor: float = 1.0e-3,
        fbd_k_factor: float = 5.0e-4,
        fbd_rho: float = 2330.0,
        inp_cp_factor: float = 8.0e-4,
        inp_k_factor: float = 8.0e-4,
        inp_rho: float = 4810.0,
        R_K: float = 5.0e-8,
        heat_source: Optional[Union[float, np.ndarray]] = None,
    ) -> None:
        self.shape = tuple(int(s) for s in shape)
        self.dx = self.dy = self.dz = float(spacing)
        self.T_bath = float(T_bath)
        self.R_K = float(R_K)

        # Derived geometric factors
        self.Ax = self.dy * self.dz
        self.Ay = self.dx * self.dz
        self.Az = self.dx * self.dy
        self.dV = self.dx * self.dy * self.dz

        # Material masks
        self.mask = self._build_mask(fd_soi_thickness)
        self.rho_arr = np.where(self.mask == 0, fbd_rho, inp_rho)
        self.k_prefactor = np.where(self.mask == 0, fbd_k_factor, inp_k_factor)
        self.cp_prefactor = np.where(self.mask == 0, fbd_cp_factor, inp_cp_factor)

        if heat_source is None:
            self.heat_source = np.zeros(self.shape, dtype=float)
        else:
            self.heat_source = _to_array(heat_source, self.shape)

    def _build_mask(self, fd_soi_thickness: float) -> np.ndarray:
        """Return a 3-D array where 0 = FD-SOI and 1 = InP."""
        mask = np.ones(self.shape, dtype=int)
        nz = self.shape[2]
        n_fd = max(1, min(nz, int(fd_soi_thickness / self.dz)))
        mask[:, :, :n_fd] = 0
        return mask

    def k(self, T: np.ndarray) -> np.ndarray:
        """Temperature-dependent thermal conductivity."""
        return self.k_prefactor * (T ** 3)

    def cp(self, T: np.ndarray) -> np.ndarray:
        """Temperature-dependent specific heat."""
        return self.cp_prefactor * (T ** 3)

    def volumetric_heat_capacity(self, T: np.ndarray) -> np.ndarray:
        """rho * C_p(T)  [J m^-3 K^-1]."""
        return self.rho_arr * self.cp(T)

    def thermal_diffusivity(self, T: np.ndarray) -> np.ndarray:
        """k(T) / (rho*C_p(T))  [m^2 s^-1]."""
        return self.k(T) / self.volumetric_heat_capacity(T)

    def _face_conductance(
        self,
        k1: np.ndarray,
        k2: np.ndarray,
        m1: np.ndarray,
        m2: np.ndarray,
        d: float,
        A: float,
    ) -> np.ndarray:
        """Finite-volume conductance between two adjacent cell centres."""
        R_iface = np.where(m1 != m2, self.R_K, 0.0)
        R = d / (2.0 * k1) + d / (2.0 * k2) + R_iface
        return A / R

    def _boundary_conductance(self, k_face: np.ndarray, d: float, A: float) -> np.ndarray:
        """Conductance from a surface cell to the T_bath bath through Kapitza R_K."""
        R = d / (2.0 * k_face) + self.R_K
        return A / R

    def _assemble(self, T: np.ndarray, k: np.ndarray, q: np.ndarray) -> tuple:
        """Assemble the finite-volume numerator and denominator for energy balance.

        ``num`` = sum(G_ij * T_j) + G_bath * T_bath + q * dV
        ``den`` = sum(G_ij) + G_bath

        The steady-state temperature is num/den; for a transient step,
        dT/dt = (num - den * T) / (rho*C_p*dV).
        """
        nx, ny, nz = self.shape
        num = q * self.dV
        den = np.zeros_like(T)

        # x-direction faces
        Gx = self._face_conductance(k[:-1, :, :], k[1:, :, :],
                                    self.mask[:-1, :, :], self.mask[1:, :, :],
                                    self.dx, self.Ax)
        num[:nx - 1, :, :] += Gx * T[1:, :, :]
        num[1:, :, :] += Gx * T[:-1, :, :]
        den[:nx - 1, :, :] += Gx
        den[1:, :, :] += Gx
        Gx0 = self._boundary_conductance(k[0, :, :], self.dx, self.Ax)
        GxN = self._boundary_conductance(k[-1, :, :], self.dx, self.Ax)
        num[0, :, :] += Gx0 * self.T_bath
        num[-1, :, :] += GxN * self.T_bath
        den[0, :, :] += Gx0
        den[-1, :, :] += GxN

        # y-direction faces
        Gy = self._face_conductance(k[:, :-1, :], k[:, 1:, :],
                                    self.mask[:, :-1, :], self.mask[:, 1:, :],
                                    self.dy, self.Ay)
        num[:, :ny - 1, :] += Gy * T[:, 1:, :]
        num[:, 1:, :] += Gy * T[:, :-1, :]
        den[:, :ny - 1, :] += Gy
        den[:, 1:, :] += Gy
        Gy0 = self._boundary_conductance(k[:, 0, :], self.dy, self.Ay)
        GyN = self._boundary_conductance(k[:, -1, :], self.dy, self.Ay)
        num[:, 0, :] += Gy0 * self.T_bath
        num[:, -1, :] += GyN * self.T_bath
        den[:, 0, :] += Gy0
        den[:, -1, :] += GyN

        # z-direction faces
        Gz = self._face_conductance(k[:, :, :-1], k[:, :, 1:],
                                    self.mask[:, :, :-1], self.mask[:, :, 1:],
                                    self.dz, self.Az)
        num[:, :, :nz - 1] += Gz * T[:, :, 1:]
        num[:, :, 1:] += Gz * T[:, :, :-1]
        den[:, :, :nz - 1] += Gz
        den[:, :, 1:] += Gz
        Gz0 = self._boundary_conductance(k[:, :, 0], self.dz, self.Az)
        GzN = self._boundary_conductance(k[:, :, -1], self.dz, self.Az)
        num[:, :, 0] += Gz0 * self.T_bath
        num[:, :, -1] += GzN * self.T_bath
        den[:, :, 0] += Gz0
        den[:, :, -1] += GzN

        return num, den

    def solve_steady_state(
        self,
        q: Optional[Union[float, np.ndarray]] = None,
        max_iter: int = 3000,
        tol: float = 1e-7,
        omega: float = 0.65,
    ) -> np.ndarray:
        """Picard/fixed-point finite-volume solve for steady-state temperature.

        Parameters
        ----------
        q
            Volumetric heat source [W m^-3].  If ``None``, ``self.heat_source`` is used.
        max_iter, tol, omega
            Nonlinear iteration controls.  ``omega`` is a relaxation factor.
        """
        if q is None:
            q = self.heat_source
        q_arr = _to_array(q, self.shape)

        T = np.full(self.shape, self.T_bath, dtype=float)
        for _ in range(max_iter):
            k = self.k(T)
            num, den = self._assemble(T, k, q_arr)
            T_new = (1.0 - omega) * T + omega * (num / den)
            T_new = np.maximum(T_new, self.T_bath)
            err = float(np.max(np.abs(T_new - T)))
            T = T_new
            if err < tol:
                break
        return T

    def solve_transient(
        self,
        n_steps: int = 200,
        q: Optional[Union[float, np.ndarray]] = None,
        dt: Optional[float] = None,
        cfl: float = 0.2,
    ) -> tuple:
        """Explicit finite-volume thermal diffusion for ``n_steps``.

        Returns the final temperature array and the simulated elapsed time.
        """
        if q is None:
            q = self.heat_source
        q_arr = _to_array(q, self.shape)

        T = np.full(self.shape, self.T_bath, dtype=float)

        if dt is None:
            alpha = self.thermal_diffusivity(T)
            alpha_max = float(np.max(alpha[np.isfinite(alpha)]))
            min_spacing = min(self.dx, self.dy, self.dz)
            dt = cfl * (min_spacing ** 2) / (6.0 * alpha_max + 1e-30)

        for _ in range(n_steps):
            k = self.k(T)
            rhoc = self.volumetric_heat_capacity(T)
            num, den = self._assemble(T, k, q_arr)
            dTdt = (num - den * T) / (rhoc * self.dV)
            T = T + dt * dTdt
            T = np.maximum(T, self.T_bath)
        return T, float(n_steps * dt)

    def kapitza_temperature_jump(self, T: np.ndarray, axis: int = 2) -> float:
        """Mean temperature jump across the FD-SOI / InP interface due to R_K."""
        nz = self.shape[2]
        n_fd = int(np.sum(self.mask[:, :, 0] == 0))  # not used
        # Locate the interface along the chosen axis
        if axis == 2:
            iface = max(1, nz // 2)
            T_below = T[:, :, iface - 1]
            T_above = T[:, :, iface]
        else:
            raise NotImplementedError("Only z-axis interface is currently supported")
        return float(np.mean(np.abs(T_above - T_below)))


class VibrationJitterFEA:
    """Mechanical vibration spectrum and optical waveguide phase-jitter model.

    The vibration environment is represented by an acceleration power spectral
    density S_a(f) in the 1 Hz - 500 Hz band.  Acceleration is double-integrated
    to displacement:

        H_a2d(f) = -1 / (2*pi*f)^2   [m / (m s^-2)]

    A piezo feed-forward cancellation path is modelled by a band-limited copy
    of the acceleration-to-displacement transfer function:

        H_ff(f) = K_ff * H_a2d(f) / (1 + j f/f_c)

    The residual displacement spectrum is |H_a2d - H_ff|^2 * S_a.  The resulting
    optical phase-noise PSD is

        S_phi(f) = (2*pi*n_eff / wavelength)^2 * S_x,res(f)

    and the RMS phase jitter is sqrt(int S_phi df).
    """

    def __init__(
        self,
        freqs: Optional[np.ndarray] = None,
        waveguide_length: float = 0.05,
        n_eff: float = 1.5,
        wavelength: float = 1.55e-6,
        base_accel_psd: float = 1.0e-9,
        resonance_freqs: Optional[Sequence[float]] = None,
        resonance_amps: Optional[Sequence[float]] = None,
        resonance_q: float = 10.0,
    ) -> None:
        if freqs is None:
            freqs = np.linspace(1.0, 500.0, 512)
        self.freqs = np.asarray(freqs, dtype=float)
        if np.any(self.freqs <= 0):
            raise ValueError("Frequencies must be positive")

        self.L = float(waveguide_length)
        self.n_eff = float(n_eff)
        self.wavelength = float(wavelength)
        self.base_accel_psd = float(base_accel_psd)
        self.resonance_q = float(resonance_q)

        if resonance_freqs is None:
            resonance_freqs = [55.0, 120.0, 245.0]
        if resonance_amps is None:
            resonance_amps = [2.0e-9, 1.0e-9, 5.0e-10]
        self.resonance_freqs = np.asarray(resonance_freqs, dtype=float)
        self.resonance_amps = np.asarray(resonance_amps, dtype=float)

        self.optical_gain = (2.0 * math.pi * self.n_eff) / self.wavelength

    def acceleration_psd(self) -> np.ndarray:
        """Acceleration power spectral density [(m/s^2)^2 / Hz]."""
        f = self.freqs
        # Coloured background: mid-band emphasis, rolling off at high frequency
        S = self.base_accel_psd * (
            1.0 + (f / 50.0) ** 2
        ) / (
            1.0 + (f / 200.0) ** 4
        )
        # Add Lorentzian mechanical resonances
        for fr, amp in zip(self.resonance_freqs, self.resonance_amps):
            S += amp * fr ** 2 / ((f ** 2 - fr ** 2) ** 2 + (fr * f / self.resonance_q) ** 2)
        return S

    def acceleration_to_displacement_tf(self) -> np.ndarray:
        """Mechanical transfer function from acceleration to displacement."""
        return -1.0 / ((2.0 * math.pi * self.freqs) ** 2)

    def piezo_feedforward_tf(self, K_ff: float = 1.0, f_c: float = 100.0) -> np.ndarray:
        """Piezo feed-forward cancellation transfer function.

        K_ff is the feed-forward gain and f_c is the actuator/control bandwidth.
        """
        f = self.freqs
        H_a2d = self.acceleration_to_displacement_tf()
        return K_ff * H_a2d / (1.0 + 1j * f / f_c)

    def displacement_psd(
        self,
        K_ff: float = 0.0,
        f_c: float = 100.0,
    ) -> np.ndarray:
        """Residual displacement PSD after optional piezo feed-forward cancellation."""
        S_a = self.acceleration_psd()
        H_a2d = self.acceleration_to_displacement_tf()
        H_ff = self.piezo_feedforward_tf(K_ff=K_ff, f_c=f_c)
        residual = H_a2d - H_ff
        return S_a * (np.abs(residual) ** 2)

    def phase_noise_psd(self, K_ff: float = 0.0, f_c: float = 100.0) -> np.ndarray:
        """Optical phase-noise PSD [rad^2 / Hz]."""
        S_x = self.displacement_psd(K_ff=K_ff, f_c=f_c)
        return self.optical_gain ** 2 * S_x

    def rms_phase_jitter(self, K_ff: float = 0.0, f_c: float = 100.0) -> float:
        """Integrated RMS phase jitter [rad]."""
        S_phi = self.phase_noise_psd(K_ff=K_ff, f_c=f_c)
        return float(np.sqrt(np.trapezoid(S_phi, self.freqs)))

    def jitter_reduction_db(self, K_ff: float = 1.0, f_c: float = 100.0) -> float:
        """RMS phase-jitter reduction relative to the open-loop case [dB]."""
        open_loop = self.rms_phase_jitter(K_ff=0.0, f_c=f_c)
        cancelled = self.rms_phase_jitter(K_ff=K_ff, f_c=f_c)
        return float(-20.0 * math.log10(cancelled / (open_loop + 1e-30)))


class RogersInterposerSIPISolver:
    """Signal and power-integrity solver for a 12-layer Rogers RO4350B interposer.

    Microstrip models (effective dielectric constant, characteristic impedance,
    attenuation) are combined with weak-coupling FEXT and a lumped PDN model to
    evaluate high-frequency signal and power behaviour.

    Attributes
    ----------
    num_layers : int
        Number of RO4350B layers (default 12).
    layer_thickness : float
        Thickness of each dielectric layer [m].
    trace_width, trace_thickness, trace_length : float
        Microstrip geometry [m].
    er, tan_delta : float
        RO4350B dielectric permittivity and loss tangent.
    """

    def __init__(
        self,
        num_layers: int = 12,
        layer_thickness: float = 0.1e-3,
        trace_width: float = 0.2e-3,
        trace_thickness: float = 18.0e-6,
        trace_length: float = 10.0e-3,
        line_spacing: float = 0.3e-3,
        er: float = 3.66,
        tan_delta: float = 0.0031,
        copper_sigma: float = 5.8e7,
        c_m_ratio: float = 0.12,
        l_m_ratio: float = 0.08,
        decap_inventory: Optional[List[dict]] = None,
        plane_cap: float = 20.0e-9,
        plane_esl: float = 0.05e-9,
        plane_esr: float = 0.001,
        vrm_r: float = 0.005,
        vrm_l: float = 0.2e-9,
    ) -> None:
        self.num_layers = int(num_layers)
        self.layer_thickness = float(layer_thickness)
        self.trace_width = float(trace_width)
        self.trace_thickness = float(trace_thickness)
        self.trace_length = float(trace_length)
        self.line_spacing = float(line_spacing)
        self.er = float(er)
        self.tan_delta = float(tan_delta)
        self.copper_sigma = float(copper_sigma)
        self.c_m_ratio = float(c_m_ratio)
        self.l_m_ratio = float(l_m_ratio)
        self.plane_cap = float(plane_cap)
        self.plane_esl = float(plane_esl)
        self.plane_esr = float(plane_esr)
        self.vrm_r = float(vrm_r)
        self.vrm_l = float(vrm_l)

        if decap_inventory is None:
            # Default per-layer decoupling banks; the inventory is interpreted as
            # one set per layer and then scaled by the number of layers.
            base = [
                {"c": 4.7e-6, "esl": 0.5e-9, "esr": 0.010},
                {"c": 100.0e-9, "esl": 0.2e-9, "esr": 0.015},
                {"c": 10.0e-9, "esl": 0.1e-9, "esr": 0.030},
                {"c": 1.0e-9, "esl": 0.05e-9, "esr": 0.080},
            ]
            self.decap_inventory = []
            for cap in base:
                self.decap_inventory.append(
                    {
                        "c": cap["c"] * self.num_layers,
                        "esl": cap["esl"] / max(1, self.num_layers),
                        "esr": cap["esr"] / max(1, self.num_layers),
                    }
                )
        else:
            self.decap_inventory = list(decap_inventory)

    def _w_over_h(self) -> float:
        return self.trace_width / self.layer_thickness

    def effective_dielectric_constant(self) -> float:
        """Hammerstad-Jensen closed-form effective permittivity."""
        u = self._w_over_h()
        if u <= 1.0:
            F = 1.0 / math.sqrt(1.0 + 12.0 / u) + 0.04 * (1.0 - u) ** 2
        else:
            F = 1.0 / math.sqrt(1.0 + 12.0 / u) + 0.04 * (1.0 - u)
        eps_eff = (self.er + 1.0) / 2.0 + (self.er - 1.0) / 2.0 * F
        return float(eps_eff)

    def characteristic_impedance(self) -> float:
        """Wheeler/Hammerstad microstrip characteristic impedance."""
        u = self._w_over_h()
        eps_eff = self.effective_dielectric_constant()
        if u <= 1.0:
            Z0 = (60.0 / math.sqrt(eps_eff)) * math.log(8.0 / u + u / 4.0)
        else:
            Z0 = (
                120.0 * math.pi
            ) / (
                math.sqrt(eps_eff)
                * (u + 1.393 + 0.667 * math.log(u + 1.444))
            )
        return float(Z0)

    def phase_velocity(self) -> float:
        """Signal phase velocity in the microstrip."""
        return C0 / math.sqrt(self.effective_dielectric_constant())

    def s21(self, frequency: np.ndarray) -> np.ndarray:
        """Return complex S21 for a single microstrip of length ``trace_length``."""
        f = np.asarray(frequency, dtype=float)
        eps_eff = self.effective_dielectric_constant()
        Z0 = self.characteristic_impedance()

        # Propagation constant
        beta = 2.0 * math.pi * f * math.sqrt(eps_eff) / C0

        # Dielectric attenuation [Np/m]
        alpha_d = (math.pi * f / C0) * math.sqrt(eps_eff) * self.tan_delta

        # Conductor attenuation [Np/m]
        R_s = np.sqrt(math.pi * f * MU0 / self.copper_sigma)
        alpha_c = R_s / (Z0 * self.trace_width)

        alpha_total = alpha_d + alpha_c
        gamma = alpha_total + 1j * beta
        return np.exp(-gamma * self.trace_length)

    def insertion_loss_db(self, frequency: np.ndarray) -> np.ndarray:
        """S21 insertion loss in dB."""
        return 20.0 * np.log10(np.maximum(1e-30, np.abs(self.s21(frequency))))

    def fext(self, frequency: np.ndarray) -> np.ndarray:
        """Far-End Crosstalk (FEXT) of two adjacent microstrips [dB].

        A weakly-coupled, frequency-domain approximation is used:

            V_FEXT/V_in ~ 0.5 * (C_m/C_11 - L_m/L_11) * (2*pi*f*l / v_p)
        """
        f = np.asarray(frequency, dtype=float)
        v_p = self.phase_velocity()
        coupling_diff = self.c_m_ratio - self.l_m_ratio
        # Electrical length term
        el = 2.0 * math.pi * f * self.trace_length / v_p
        linear = 0.5 * abs(coupling_diff) * el
        # Cap the linear coupling model to avoid unphysical >0 dB values
        linear = np.minimum(linear, 0.999)
        safe = np.where(f > 0.0, linear, 1e-15)
        return 20.0 * np.log10(np.maximum(1e-15, safe))

    def _plane_admittance(self, omega: np.ndarray) -> np.ndarray:
        """Admittance of the parallel plane capacitance."""
        n_pairs = max(1, self.num_layers - 1)
        C = self.plane_cap * n_pairs
        L = self.plane_esl / n_pairs
        R = self.plane_esr / n_pairs
        # Series RLC admittance: Y = j*omega*C / (1 - omega^2*L*C + j*omega*C*R)
        denom = 1.0 - omega ** 2 * L * C + 1j * omega * C * R
        return 1j * omega * C / denom

    def z_pdn(self, frequency: np.ndarray) -> np.ndarray:
        """Power Distribution Network impedance Z_PDN(f) [Ohm].

        The model is a parallel combination of the VRM output impedance, the
        aggregate plane capacitance, and the scaled decoupling-capacitor banks.
        """
        f = np.asarray(frequency, dtype=float)
        omega = 2.0 * math.pi * f

        # VRM branch
        Z_vrm = self.vrm_r + 1j * omega * self.vrm_l
        Y_total = 1.0 / Z_vrm

        # Plane pair
        Y_total += self._plane_admittance(omega)

        # Decoupling capacitors (series RLC admittance)
        for cap in self.decap_inventory:
            C = cap["c"]
            L = cap["esl"]
            R = cap["esr"]
            denom = 1.0 - omega ** 2 * L * C + 1j * omega * C * R
            Y_total += 1j * omega * C / denom

        Z = 1.0 / Y_total
        # DC guarantee
        if np.any(f == 0.0):
            Z = np.where(f == 0.0, self.vrm_r + 0.0j, Z)
        return Z

    def si_pi_summary(self, frequency: np.ndarray) -> dict:
        """Convenience bundle of S21, FEXT, and Z_PDN for a given frequency vector."""
        return {
            "frequency": np.asarray(frequency),
            "s21": self.s21(frequency),
            "fext_db": self.fext(frequency),
            "z_pdn": self.z_pdn(frequency),
        }


def run_multiphysics_tests() -> None:
    """Run self-contained assertion tests for the three physics solvers."""
    print("Running SHBT-R multi-physics self-tests...")

    # ------------------------------------------------------------------
    # 1. CryogenicThermalSolver
    # ------------------------------------------------------------------
    thermal = CryogenicThermalSolver(
        shape=(12, 12, 12),
        spacing=5.0e-9,
        T_bath=4.2,
        fd_soi_thickness=28.0e-9,
        R_K=5.0e-8,
    )
    q = np.zeros(thermal.shape, dtype=float)
    # Localised heat source in the FD-SOI layer
    q[3:6, 4:8, 1:3] = 1.0e14  # W/m^3

    T_ss = thermal.solve_steady_state(q=q, max_iter=2000, tol=1e-6)
    assert T_ss.shape == thermal.shape, "Thermal steady-state shape mismatch"
    assert np.all(np.isfinite(T_ss)), "Non-finite temperatures"
    assert float(np.max(T_ss)) > thermal.T_bath + 0.05, "Heat source should raise temperature above bath"
    assert float(np.min(T_ss)) >= thermal.T_bath - 1e-6, "Temperature below bath is non-physical"

    T_tr, t_final = thermal.solve_transient(n_steps=100, q=q)
    assert T_tr.shape == thermal.shape, "Transient thermal shape mismatch"
    assert np.all(np.isfinite(T_tr)), "Non-finite transient temperatures"
    assert float(np.max(T_tr)) > thermal.T_bath, "Transient heating must occur"
    assert float(t_final) > 0.0, "Positive simulation time"

    jump = thermal.kapitza_temperature_jump(T_ss)
    print(f"  Thermal: max T = {np.max(T_ss):.4f} K, Kapitza jump = {jump:.6f} K, t_final = {t_final:.3e} s")

    # ------------------------------------------------------------------
    # 2. VibrationJitterFEA
    # ------------------------------------------------------------------
    freqs = np.linspace(1.0, 500.0, 400)
    vib = VibrationJitterFEA(
        freqs=freqs,
        waveguide_length=0.05,
        n_eff=1.5,
        wavelength=1.55e-6,
    )
    rms_open = vib.rms_phase_jitter(K_ff=0.0, f_c=100.0)
    rms_cancel = vib.rms_phase_jitter(K_ff=1.0, f_c=100.0)
    assert rms_open > 0.0, "Open-loop RMS phase jitter must be positive"
    assert rms_cancel > 0.0, "Cancelled RMS phase jitter must be positive"
    assert rms_cancel < 0.95 * rms_open, "Piezo feed-forward should reduce RMS phase jitter"
    reduction_db = vib.jitter_reduction_db(K_ff=1.0, f_c=100.0)
    print(f"  Vibration: open-loop = {rms_open:.3e} rad, cancelled = {rms_cancel:.3e} rad, reduction = {reduction_db:.2f} dB")

    # ------------------------------------------------------------------
    # 3. RogersInterposerSIPISolver
    # ------------------------------------------------------------------
    interposer = RogersInterposerSIPISolver(
        num_layers=12,
        layer_thickness=0.1e-3,
        trace_width=0.2e-3,
        trace_length=10.0e-3,
    )
    f_rf = np.logspace(9.0, 11.0, 60)  # 1 GHz to 100 GHz for S21/FEXT
    f_pdn = np.logspace(6.0, 9.0, 60)  # 1 MHz to 1 GHz for Z_PDN decap effectiveness
    s21 = interposer.s21(f_rf)
    fext_db = interposer.fext(f_rf)
    z_pdn = interposer.z_pdn(f_pdn)

    assert s21.size == f_rf.size, "S21 frequency-vector size mismatch"
    assert fext_db.size == f_rf.size, "FEXT frequency-vector size mismatch"
    assert z_pdn.size == f_pdn.size, "Z_PDN frequency-vector size mismatch"
    assert np.all(np.isfinite(s21)), "Non-finite S21"
    assert np.all(np.isfinite(fext_db)), "Non-finite FEXT"
    assert np.all(np.isfinite(z_pdn)), "Non-finite Z_PDN"
    assert np.all(np.abs(s21) <= 1.0 + 1e-9), "Passive S21 magnitude must be <= 1"
    # S21 magnitude should decrease monotonically with frequency for this model
    assert np.all(np.abs(s21)[1:] <= np.abs(s21)[:-1]), "S21 magnitude should decrease with frequency"
    assert fext_db[-1] > fext_db[0], "FEXT should increase (become less negative) with frequency"
    assert np.all(fext_db <= 0.0 + 1e-9), "FEXT should be <= 0 dB"
    assert np.all(np.real(z_pdn) > 0.0), "PDN real impedance must be positive"
    assert float(np.min(np.abs(z_pdn))) < interposer.vrm_r, "Decoupling must create an impedance below the VRM at some frequency"

    print(f"  Interposer: S21(10 GHz) = {20*np.log10(np.abs(s21[20])):.2f} dB, FEXT(10 GHz) = {fext_db[20]:.2f} dB")

    print("All multi-physics self-tests passed.")


if __name__ == "__main__":
    run_multiphysics_tests()
