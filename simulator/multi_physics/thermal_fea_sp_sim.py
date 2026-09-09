"""Cryogenic thermal FEA for the SHBT-R quantum-computing substrate.

Models a 3-D finite-volume stack containing the SOI microcavity, a
nanoporous aerogel matching layer, and a sapphire substrate at 4.2 K.
Material properties are temperature dependent:

    C_v(T) = gamma_1 T + beta_3 T^3
    k(T)   = kappa_0 T^alpha

Kapitza interfacial thermal resistance is included between dissimilar
materials and at the bath boundary:

    R_K = 1 / (alpha_K T^3)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np


@dataclass
class Material:
    """Thermal material definition for the cryogenic stack."""

    name: str
    rho: float          # kg/m^3
    gamma1: float       # J/(kg K^2)
    beta3: float        # J/(kg K^4)
    kappa0: float       # W/(m K^(1+alpha))
    alpha: float        # conductivity exponent
    refractive_index: Optional[float] = None


# Nominal SHBT-R material properties (order-of-magnitude cryogenic values).
MICROCAVITY = Material(
    name="SOI microcavity",
    rho=2329.0,
    gamma1=0.01,
    beta3=1.0e-6,
    kappa0=1.0e-3,
    alpha=3.0,
    refractive_index=3.17,
)

AEROGEL = Material(
    name="nanoporous aerogel",
    rho=150.0,
    gamma1=0.001,
    beta3=1.0e-9,
    kappa0=1.0e-5,
    alpha=2.5,
    refractive_index=1.0182,
)

SAPPHIRE = Material(
    name="sapphire substrate",
    rho=3980.0,
    gamma1=0.02,
    beta3=1.0e-9,
    kappa0=1.0e-2,
    alpha=3.0,
    refractive_index=1.76,
)


def _to_array(value: Union[float, np.ndarray], shape: Tuple[int, ...]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape == ():
        return np.full(shape, float(value), dtype=float)
    if arr.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {arr.shape}")
    return arr


class CryogenicThermalSolver:
    """Non-linear 3-D finite-volume thermal solver with Kapitza interfaces."""

    def __init__(
        self,
        shape: Tuple[int, int, int] = (12, 12, 12),
        spacing: float = 5.0e-9,
        T_bath: float = 4.2,
        microcavity_thickness: float = 50.0e-9,
        aerogel_thickness: float = 6.395e-9,
        microcavity: Material = MICROCAVITY,
        aerogel: Material = AEROGEL,
        sapphire: Material = SAPPHIRE,
        alpha_K: float = 142.0,
        heat_source: Optional[Union[float, np.ndarray]] = None,
    ):
        self.shape = tuple(int(s) for s in shape)
        self.dx = self.dy = self.dz = float(spacing)
        self.T_bath = float(T_bath)
        self.alpha_K = float(alpha_K)

        self.Ax = self.dy * self.dz
        self.Ay = self.dx * self.dz
        self.Az = self.dx * self.dy
        self.dV = self.dx * self.dy * self.dz

        self.materials = [microcavity, aerogel, sapphire]
        self.microcavity = microcavity
        self.aerogel = aerogel
        self.sapphire = sapphire

        self.mask = self._build_mask(microcavity_thickness, aerogel_thickness)
        self.rho_arr = self._prop("rho")
        self.gamma1_arr = self._prop("gamma1")
        self.beta3_arr = self._prop("beta3")
        self.kappa0_arr = self._prop("kappa0")
        self.alpha_arr = self._prop("alpha")

        self.heat_source = _to_array(
            heat_source if heat_source is not None else 0.0, self.shape
        )

    def _build_mask(
        self, t_cav: float, t_aero: float
    ) -> np.ndarray:
        nz = self.shape[2]
        mask = np.full(self.shape, 2, dtype=int)  # default sapphire
        n_cav = max(1, min(nz, int(math.ceil(t_cav / self.dz))))
        n_aero = max(0, min(nz - n_cav, int(math.ceil(t_aero / self.dz))))
        mask[:, :, :n_cav] = 0  # microcavity
        mask[:, :, n_cav : n_cav + n_aero] = 1  # aerogel
        return mask

    def _prop(self, attr: str) -> np.ndarray:
        arr = np.empty(self.shape, dtype=float)
        for idx, mat in enumerate(self.materials):
            arr[self.mask == idx] = getattr(mat, attr)
        return arr

    def k(self, T: np.ndarray) -> np.ndarray:
        """Temperature-dependent thermal conductivity [W/(m K)]."""
        return self.kappa0_arr * (T ** self.alpha_arr)

    def cp(self, T: np.ndarray) -> np.ndarray:
        """Temperature-dependent specific heat [J/(kg K)]."""
        return self.gamma1_arr * T + self.beta3_arr * (T ** 3)

    def volumetric_heat_capacity(self, T: np.ndarray) -> np.ndarray:
        return self.rho_arr * self.cp(T)

    def thermal_diffusivity(self, T: np.ndarray) -> np.ndarray:
        return self.k(T) / self.volumetric_heat_capacity(T)

    def _kapitza_resistance(self, T: np.ndarray) -> np.ndarray:
        """Kapitza interfacial resistance [m^2 K / W]."""
        return 1.0 / (self.alpha_K * (T ** 3))

    def _face_conductance(
        self,
        k1: np.ndarray,
        k2: np.ndarray,
        m1: np.ndarray,
        m2: np.ndarray,
        T1: np.ndarray,
        T2: np.ndarray,
        d: float,
        A: float,
    ) -> np.ndarray:
        """Thermal conductance between two neighboring control volumes."""
        R_cond = d / (2.0 * k1) + d / (2.0 * k2)
        T_face = 0.5 * (T1 + T2)
        R_k = np.where(m1 != m2, self._kapitza_resistance(T_face), 0.0)
        return A / (R_cond + R_k)

    def _boundary_conductance(
        self, k_face: np.ndarray, T_face: np.ndarray, d: float, A: float
    ) -> np.ndarray:
        """Thermal conductance from a boundary cell to the 4.2 K bath."""
        R_cond = d / (2.0 * k_face)
        R_k = self._kapitza_resistance(T_face)
        return A / (R_cond + R_k)

    def _assemble(self, T: np.ndarray, k: np.ndarray, q: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble finite-volume numerator and denominator for T = num/den."""
        nx, ny, nz = self.shape
        num = q * self.dV
        den = np.zeros_like(T)

        # x-direction
        Gx = self._face_conductance(
            k[:-1], k[1:],
            self.mask[:-1], self.mask[1:],
            T[:-1], T[1:],
            self.dx, self.Ax,
        )
        num[: nx - 1] += Gx * T[1:]
        num[1:] += Gx * T[:-1]
        den[: nx - 1] += Gx
        den[1:] += Gx
        Gx0 = self._boundary_conductance(k[0], T[0], self.dx, self.Ax)
        GxN = self._boundary_conductance(k[-1], T[-1], self.dx, self.Ax)
        num[0] += Gx0 * self.T_bath
        num[-1] += GxN * self.T_bath
        den[0] += Gx0
        den[-1] += GxN

        # y-direction
        Gy = self._face_conductance(
            k[:, :-1], k[:, 1:],
            self.mask[:, :-1], self.mask[:, 1:],
            T[:, :-1], T[:, 1:],
            self.dy, self.Ay,
        )
        num[:, : ny - 1] += Gy * T[:, 1:]
        num[:, 1:] += Gy * T[:, :-1]
        den[:, : ny - 1] += Gy
        den[:, 1:] += Gy
        Gy0 = self._boundary_conductance(k[:, 0], T[:, 0], self.dy, self.Ay)
        GyN = self._boundary_conductance(k[:, -1], T[:, -1], self.dy, self.Ay)
        num[:, 0] += Gy0 * self.T_bath
        num[:, -1] += GyN * self.T_bath
        den[:, 0] += Gy0
        den[:, -1] += GyN

        # z-direction
        Gz = self._face_conductance(
            k[:, :, :-1], k[:, :, 1:],
            self.mask[:, :, :-1], self.mask[:, :, 1:],
            T[:, :, :-1], T[:, :, 1:],
            self.dz, self.Az,
        )
        num[:, :, : nz - 1] += Gz * T[:, :, 1:]
        num[:, :, 1:] += Gz * T[:, :, :-1]
        den[:, :, : nz - 1] += Gz
        den[:, :, 1:] += Gz
        Gz0 = self._boundary_conductance(k[:, :, 0], T[:, :, 0], self.dz, self.Az)
        GzN = self._boundary_conductance(k[:, :, -1], T[:, :, -1], self.dz, self.Az)
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
        """Solve the non-linear steady-state thermal field."""
        q_arr = _to_array(q if q is not None else self.heat_source, self.shape)
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
    ) -> Tuple[np.ndarray, float]:
        """Integrate the transient thermal response with explicit Euler."""
        q_arr = _to_array(q if q is not None else self.heat_source, self.shape)
        T = np.full(self.shape, self.T_bath, dtype=float)
        if dt is None:
            alpha = self.thermal_diffusivity(T)
            finite = alpha[np.isfinite(alpha)]
            alpha_max = float(np.max(finite)) if finite.size else 1.0
            min_spacing = min(self.dx, self.dy, self.dz)
            dt = cfl * (min_spacing ** 2) / (6.0 * alpha_max + 1e-30)
        for _ in range(int(n_steps)):
            k = self.k(T)
            rhoc = self.volumetric_heat_capacity(T)
            num, den = self._assemble(T, k, q_arr)
            dTdt = (num - den * T) / (rhoc * self.dV)
            T = T + dt * dTdt
            T = np.maximum(T, self.T_bath)
        return T, float(int(n_steps) * dt)

    def simulate_laser_quench(
        self,
        peak_power_density: float = 1.0e15,
        duration: float = 1.0e-12,
        beam_sigma: Tuple[float, float, float] = (1.0e-6, 1.0e-6, 5.0e-9),
        center: Optional[Tuple[int, int, int]] = None,
    ) -> Tuple[np.ndarray, float]:
        """Simulate a transient Gaussian laser thermal quench in the microcavity."""
        if center is None:
            center = (self.shape[0] // 2, self.shape[1] // 2, 1)

        sigma_cells = (
            max(1.0, beam_sigma[0] / self.dx),
            max(1.0, beam_sigma[1] / self.dy),
            max(1.0, beam_sigma[2] / self.dz),
        )
        ix, iy, iz = np.indices(self.shape, dtype=float)
        cx, cy, cz = center

        gaussian = np.exp(
            -((ix - cx) / sigma_cells[0]) ** 2
            - ((iy - cy) / sigma_cells[1]) ** 2
            - ((iz - cz) / sigma_cells[2]) ** 2
        )
        # Laser deposits energy only inside the microcavity layer.
        microcavity_mask = self.mask == 0
        q = np.where(microcavity_mask, peak_power_density * gaussian, 0.0)

        alpha = self.thermal_diffusivity(np.full(self.shape, self.T_bath))
        finite = alpha[np.isfinite(alpha)]
        alpha_max = float(np.max(finite)) if finite.size else 1.0
        min_spacing = min(self.dx, self.dy, self.dz)
        dt = 0.2 * (min_spacing ** 2) / (6.0 * alpha_max + 1e-30)
        n_steps = max(1, int(np.ceil(duration / dt)))

        return self.solve_transient(n_steps=n_steps, q=q, dt=dt)

    def kapitza_temperature_jump(self, T: np.ndarray) -> float:
        """Return the mean Kapitza temperature jump across material interfaces."""
        jumps = []
        T = np.asarray(T)
        # x interfaces
        diffs_x = self.mask[:-1] != self.mask[1:]
        if np.any(diffs_x):
            T_avg_x = 0.5 * (T[:-1] + T[1:])[diffs_x]
            q_x = (
                self.alpha_K * (T_avg_x ** 3) * np.abs(T[:-1] - T[1:])[diffs_x]
            )
            R_K_x = 1.0 / (self.alpha_K * (T_avg_x ** 3))
            jumps.append(np.mean(R_K_x * q_x))
        # y interfaces
        diffs_y = self.mask[:, :-1] != self.mask[:, 1:]
        if np.any(diffs_y):
            T_avg_y = 0.5 * (T[:, :-1] + T[:, 1:])[diffs_y]
            q_y = (
                self.alpha_K * (T_avg_y ** 3) * np.abs(T[:, :-1] - T[:, 1:])[diffs_y]
            )
            R_K_y = 1.0 / (self.alpha_K * (T_avg_y ** 3))
            jumps.append(np.mean(R_K_y * q_y))
        # z interfaces
        diffs_z = self.mask[:, :, :-1] != self.mask[:, :, 1:]
        if np.any(diffs_z):
            T_avg_z = 0.5 * (T[:, :, :-1] + T[:, :, 1:])[diffs_z]
            q_z = (
                self.alpha_K * (T_avg_z ** 3) * np.abs(T[:, :, :-1] - T[:, :, 1:])[diffs_z]
            )
            R_K_z = 1.0 / (self.alpha_K * (T_avg_z ** 3))
            jumps.append(np.mean(R_K_z * q_z))

        return float(np.mean(jumps)) if jumps else 0.0


def run_multiphysics_tests() -> None:
    """Self-test entry point for the thermal FEA solver."""
    print("Running SHBT-R cryogenic thermal FEA self-tests...")
    solver = CryogenicThermalSolver(
        shape=(12, 12, 12),
        spacing=5.0e-9,
        T_bath=4.2,
        microcavity_thickness=50.0e-9,
        aerogel_thickness=6.395e-9,
        alpha_K=142.0,
    )

    # Localised laser-like heat source restricted to the microcavity.
    q = np.zeros(solver.shape)
    cx, cy, cz = 6, 6, 1
    q[cx - 1 : cx + 2, cy - 1 : cy + 2, 0:2] = 1.0e15

    T_ss = solver.solve_steady_state(q=q, max_iter=5000, tol=1e-8)
    assert T_ss.shape == solver.shape, "shape mismatch"
    assert np.all(np.isfinite(T_ss)), "non-finite temperatures in steady-state solution"
    assert float(np.max(T_ss)) > solver.T_bath + 0.05, "steady-state temperature did not rise above bath"
    assert float(np.min(T_ss)) >= solver.T_bath - 0.01, "temperature below bath"

    T_tr, t_final = solver.solve_transient(n_steps=400, q=q)
    assert T_tr.shape == solver.shape, "transient shape mismatch"
    assert np.all(np.isfinite(T_tr)), "non-finite transient temperatures"
    assert float(np.max(T_tr)) > solver.T_bath, "transient temperature did not rise above bath"
    assert t_final > 0.0, "non-positive time step accumulation"

    T_quench, t_quench = solver.simulate_laser_quench(
        peak_power_density=1.0e15,
        duration=1.0e-12,
    )
    assert T_quench.shape == solver.shape
    assert np.all(np.isfinite(T_quench))
    assert float(np.max(T_quench)) > solver.T_bath

    jump = solver.kapitza_temperature_jump(T_ss)
    print(f"  Thermal: max T = {np.max(T_ss):.4f} K")
    print(f"  Kapitza mean jump = {jump:.6f} K")
    print(f"  Transient t_final = {t_final:.3e} s")
    print(f"  Laser quench t_final = {t_quench:.3e} s")
    print("All SHBT-R cryogenic thermal FEA self-tests passed.")


if __name__ == "__main__":
    run_multiphysics_tests()
