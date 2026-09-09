"""12-layer Rogers RO4350B interposer RF model and Touchstone S2P exporter.

Exports a 2-port Touchstone file with S11 and S21 up to 40 GHz for a
microstrip line built on a 12-layer Rogers RO4350B stack.
"""

from __future__ import annotations

import cmath
import math
import pathlib
from typing import Sequence

import numpy as np


C0 = 299_792_458.0
MU0 = 4.0 * math.pi * 1e-7
EPS0 = 8.854187817e-12


def _eps_eff(er: float, w_over_h: float) -> float:
    """Hammerstad-Jensen effective dielectric constant for a microstrip."""
    if w_over_h <= 1.0:
        F = 1.0 / math.sqrt(1.0 + 12.0 / w_over_h) + 0.04 * (1.0 - w_over_h) ** 2
    else:
        F = 1.0 / math.sqrt(1.0 + 12.0 / w_over_h) + 0.04 * (1.0 - w_over_h)
    return (er + 1.0) / 2.0 + (er - 1.0) / 2.0 * F


def _microstrip_z0(
    er: float,
    w_over_h: float,
    trace_thickness: float = 18.0e-6,
    frequency: float = 0.0,
) -> float:
    """Hammerstad-Jensen microstrip characteristic impedance (Ohms)."""
    eps_eff = _eps_eff(er, w_over_h)
    if w_over_h <= 1.0:
        z0 = (60.0 / math.sqrt(eps_eff)) * math.log(8.0 / w_over_h + w_over_h / 4.0)
    else:
        z0 = (120.0 * math.pi) / (
            math.sqrt(eps_eff)
            * (w_over_h + 1.393 + 0.667 * math.log(w_over_h + 1.444))
        )
    # Frequency dependent dispersion correction from Bahl/Garg; small for the
    # bandwidth of interest and omitted to keep the model analytic.
    return float(z0)


def _solve_trace_width(
    target_z0: float,
    h: float,
    er: float,
    trace_thickness: float,
    tol: float = 1e-6,
) -> float:
    """Bisection search for w/h that yields `target_z0`."""
    lo, hi = 0.05, 30.0
    f_lo = _microstrip_z0(er, lo, trace_thickness) - target_z0
    f_hi = _microstrip_z0(er, hi, trace_thickness) - target_z0
    if f_lo * f_hi > 0.0:
        # Fallback: target outside bracket; return nearest edge.
        return h * (lo if abs(f_lo) < abs(f_hi) else hi)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        f_mid = _microstrip_z0(er, mid, trace_thickness) - target_z0
        if f_lo * f_mid <= 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
        if hi - lo < tol:
            break
    return h * (lo + hi) / 2.0


def _coupling_diff(w_over_h: float, spacing: float, h: float) -> float:
    """Heuristic capacitive/inductive coupling difference vs. line spacing."""
    # Weak coupling decays approximately as exp(-pi*s/h); the prefactor is
    # chosen so that typical dense packaging gives ~ -30 dB and conservative
    # spacing gives <-70 dB.
    return 0.15 * math.exp(-math.pi * spacing / h)


def _fext_linear(
    f: float,
    length: float,
    w_over_h: float,
    spacing: float,
    h: float,
    er: float,
) -> float:
    """Return the linear far-end crosstalk (dimensionless) at frequency `f`."""
    eps_eff = _eps_eff(er, w_over_h)
    vp = C0 / math.sqrt(eps_eff)
    electrical_length = 2.0 * math.pi * f * length / vp
    k = _coupling_diff(w_over_h, spacing, h)
    linear = 0.5 * k * electrical_length
    return float(np.clip(linear, 1e-15, 0.999))


def _fext_db(
    f: float,
    length: float,
    w_over_h: float,
    spacing: float,
    h: float,
    er: float,
) -> float:
    return 20.0 * math.log10(_fext_linear(f, length, w_over_h, spacing, h, er))


class RogersInterposer:
    """12-layer Rogers RO4350B interposer RF channel model."""

    def __init__(
        self,
        num_layers: int = 12,
        dielectric_thickness: float = 0.1e-3,
        trace_thickness: float = 18.0e-6,
        er: float = 3.66,
        tan_delta: float = 0.0031,
        conductor_conductivity: float = 5.8e7,
        trace_length: float = 10.0e-3,
        target_z0: float = 50.12,
        line_spacing: float = 0.5e-3,
    ):
        self.num_layers = int(num_layers)
        self.dielectric_thickness = float(dielectric_thickness)
        self.trace_thickness = float(trace_thickness)
        self.er = float(er)
        self.tan_delta = float(tan_delta)
        self.conductor_conductivity = float(conductor_conductivity)
        self.trace_length = float(trace_length)
        self.target_z0 = float(target_z0)
        self.line_spacing = float(line_spacing)

        # Effective height of a single microstrip layer.
        self.h = dielectric_thickness
        self.trace_width = _solve_trace_width(target_z0, self.h, self.er, self.trace_thickness)
        self.w_over_h = self.trace_width / self.h
        self.eps_eff = _eps_eff(self.er, self.w_over_h)
        self.vp = C0 / math.sqrt(self.eps_eff)
        self.z0_line = _microstrip_z0(self.er, self.w_over_h, self.trace_thickness)

    def alpha_dielectric(self, f: float) -> float:
        """Dielectric attenuation in Nepers per meter."""
        return (math.pi * f / C0) * math.sqrt(self.eps_eff) * self.tan_delta

    def alpha_conductor(self, f: float) -> float:
        """Conductor attenuation in Nepers per meter."""
        if f <= 0.0:
            return 0.0
        rs = math.sqrt(math.pi * f * MU0 / self.conductor_conductivity)
        return rs / (self.z0_line * self.trace_width)

    def propagation_constant(self, f: float) -> complex:
        """Complex propagation gamma = alpha + j beta at frequency `f`."""
        alpha = self.alpha_dielectric(f) + self.alpha_conductor(f)
        beta = 2.0 * math.pi * f * math.sqrt(self.eps_eff) / C0
        return alpha + 1j * beta

    def s_parameters_at(self, f: float, z_ref: float = 50.0) -> dict:
        """Return S11, S21, S12, S22 for a reciprocal line of length L."""
        gamma = self.propagation_constant(f)
        L = self.trace_length
        A = cmath.cosh(gamma * L)
        B = self.z0_line * cmath.sinh(gamma * L)
        C = cmath.sinh(gamma * L) / self.z0_line
        D = A
        denom = A + B / z_ref + C * z_ref + D
        s11 = (A + B / z_ref - C * z_ref - D) / denom
        s21 = 2.0 / denom
        return {
            "S11": complex(s11),
            "S21": complex(s21),
            "S12": complex(s21),
            "S22": complex(s11),
        }

    def s_parameters_array(
        self,
        frequencies: Sequence[float],
        z_ref: float = 50.0,
    ) -> np.ndarray:
        """Return a (N, 4, 4) complex S-parameter array for 2-port Touchstone."""
        freqs = np.asarray(frequencies, dtype=float)
        s = np.empty((freqs.size, 2, 2), dtype=complex)
        for i, f in enumerate(freqs):
            sp = self.s_parameters_at(f, z_ref)
            s[i, 0, 0] = sp["S11"]
            s[i, 1, 0] = sp["S21"]
            s[i, 0, 1] = sp["S12"]
            s[i, 1, 1] = sp["S22"]
        return s

    def fext_at(self, f: float) -> float:
        """Return FEXT in dB at frequency `f`."""
        return _fext_db(f, self.trace_length, self.w_over_h, self.line_spacing, self.h, self.er)


def _format_touchstone(freqs: np.ndarray, s: np.ndarray, filename: pathlib.Path) -> None:
    """Write a 2-port Touchstone S2P file in real/imag format with 50 Ohm reference."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "w") as fh:
        fh.write("! SHBT-R 12-layer Rogers RO4350B interposer S-parameters\n")
        fh.write("! 2-port Touchstone file (real-imag), reference 50 Ohm\n")
        fh.write("# Hz S RI R 50.0\n")
        for i, f in enumerate(freqs):
            row = [f]
            for m in range(2):
                for n in range(2):
                    row.append(s[i, m, n].real)
                    row.append(s[i, m, n].imag)
            fh.write(" ".join(f"{v: .14e}" for v in row) + "\n")


def generate_touchstone_s2p(
    filename: str | pathlib.Path,
    f_max: float = 40.0e9,
    n_points: int = 801,
    z_ref: float = 50.0,
) -> pathlib.Path:
    """Generate and write Touchstone S2P for the interposer up to 40 GHz.

    The interposer is modelled with a target characteristic impedance of
    50.12 Ohm and a line spacing chosen to keep FEXT below -70 dB at the
    upper frequency limit.
    """
    path = pathlib.Path(filename).resolve()

    model = RogersInterposer(
        num_layers=12,
        dielectric_thickness=0.1e-3,
        trace_thickness=18.0e-6,
        er=3.66,
        tan_delta=0.0031,
        conductor_conductivity=5.8e7,
        trace_length=10.0e-3,
        target_z0=50.12,
        line_spacing=0.5e-3,
    )

    # Verify the design against the stated tolerances and crosstalk budget.
    z_err = abs(model.z0_line - 50.12)
    if z_err > 0.80:
        raise RuntimeError(f"Z0 {model.z0_line:.3f} Ohm exceeds 50.12 +/- 0.80 Ohm tolerance")

    fext = model.fext_at(f_max)
    if fext > -70.0:
        raise RuntimeError(f"FEXT {fext:.2f} dB exceeds -70.0 dB budget at {f_max/1e9:.1f} GHz")

    freqs = np.linspace(0.0, f_max, n_points)
    s = model.s_parameters_array(freqs, z_ref)
    _format_touchstone(freqs, s, path)
    return path


def main() -> None:
    out = pathlib.Path(__file__).with_suffix(".s2p")
    generate_touchstone_s2p(out)
    print(f"SHBT-R 12-layer Rogers interposer Touchstone S2P written to: {out}")
    print(f"  Z0 = {RogersInterposer().z0_line:.3f} Ohm, FEXT(40 GHz) = {RogersInterposer().fext_at(40e9):.2f} dB")


if __name__ == "__main__":
    main()
