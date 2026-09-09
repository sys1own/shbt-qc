"""Wafer-ready GDSII mask layout exporter for SHBT-R hexamer PDK cells.

This module builds a C6-symmetric 6-ring hexamer cell including ring
resonators, directional couplers, and thermo-optic metal phase shifters,
and writes the layout to GDSII using `gdsfactory`.
"""

from __future__ import annotations

import math
import pathlib
from typing import Sequence, Tuple

import gdsfactory as gf
import numpy as np


def _hexagon_vertices(radius: float, n: int = 6) -> Sequence[Tuple[float, float]]:
    """Return the (x, y) coordinates of `n` vertices on a regular polygon."""
    return [(radius * math.cos(2 * math.pi * i / n), radius * math.sin(2 * math.pi * i / n)) for i in range(n)]


def hexamer_pdk_component(
    ring_radius: float = 5.0,
    ring_width: float = 0.5,
    ring_gap: float = 0.2,
    ring_pitch: float = 11.0,
    coupler_length: float = 8.0,
    coupler_gap: float = 0.2,
    heater_length: float = 10.0,
    heater_width: float = 1.0,
    waveguide_width: float = 0.5,
    layer_wg: Tuple[int, int] = (1, 0),
    layer_heater: Tuple[int, int] = (2, 0),
    layer_pad: Tuple[int, int] = (3, 0),
) -> gf.Component:
    """Return a C6-symmetric 6-ring hexamer with couplers and heaters."""
    c = gf.Component("shbt_hexamer_pdk")
    centers = _hexagon_vertices(ring_pitch / 2.0, n=6)

    # Shared waveguide cross-section (avoids creating it inside the loop)
    wg_xs = gf.cross_section.cross_section(width=waveguide_width, layer=layer_wg)

    # 6 ring resonators at hexamer vertices
    for i, (x, y) in enumerate(centers):
        ring = c << gf.components.ring(
            radius=ring_radius,
            width=ring_width,
            layer=layer_wg,
            angle_resolution=2.0,
        )
        ring.move((x, y))

    # Directional couplers: one between each adjacent pair of rings
    for i in range(6):
        x1, y1 = centers[i]
        x2, y2 = centers[(i + 1) % 6]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        coupler = c << gf.components.coupler_straight(
            length=coupler_length,
            gap=coupler_gap,
            cross_section=wg_xs,
        )
        coupler.move((mx, my))
        coupler.rotate(angle)

    # Thermo-optic phase shifter (straight waveguide + metal heater)
    ph_x = ring_pitch + 4.0
    ph_y = 0.0
    wg = c << gf.components.straight(
        length=heater_length,
        cross_section=wg_xs,
    )
    wg.move((ph_x, ph_y))

    heater = c << gf.components.rectangle(
        size=(heater_length, heater_width),
        layer=layer_heater,
        centered=True,
    )
    heater.move((ph_x, ph_y))

    # Electrical pads for the heater
    pad_w, pad_h = 4.0, 4.0
    pad_left = c << gf.components.rectangle(
        size=(pad_w, pad_h),
        layer=layer_pad,
        centered=True,
    )
    pad_left.move((ph_x - heater_length / 2.0 - pad_w, ph_y))
    pad_right = c << gf.components.rectangle(
        size=(pad_w, pad_h),
        layer=layer_pad,
        centered=True,
    )
    pad_right.move((ph_x + heater_length / 2.0 + pad_w, ph_y))

    return c


def export_gds(
    output_path: pathlib.Path | str | None = None,
    **kwargs,
) -> pathlib.Path:
    """Write the hexamer PDK layout to a GDSII file."""
    component = hexamer_pdk_component(**kwargs)
    if output_path is None:
        output_path = pathlib.Path(__file__).with_suffix(".gds")
    else:
        output_path = pathlib.Path(output_path)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    component.write_gds(output_path)
    return output_path


def main() -> None:
    out = export_gds()
    print(f"SHBT-R hexamer GDSII layout written to: {out}")


if __name__ == "__main__":
    main()
