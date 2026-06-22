"""Parametric building generator.

Applies envelope, geometry, and infiltration modifications to existing
EnergyCodes epJSON buildings, producing new building variants suitable for
cross-domain transfer benchmarks.

Intended pipeline order::

    IDF → upgrade → convert_idf → **modify_building** → add_meters → timestep → …
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from building2building.pipeline.steps.conversion import convert_idf, upgrade
from building2building.store import OUTPUT, Derivation, Realizable, derivation


@dataclass(frozen=True)
class BuildingModification:
    """A set of parametric overrides to apply to a base epJSON building.

    Every field defaults to ``None`` (no change).  Specify only the parameters
    you want to vary.
    """

    # -- Envelope (multiplicative scale on existing values) --
    envelope_conductivity_scale: float | None = None

    # -- Fenestration --
    window_u_factor: float | None = None
    window_shgc: float | None = None
    fenestration_to_wall_ratio: float | None = None

    # -- Infiltration (multiplicative scale) --
    infiltration_scale: float | None = None

    # -- Geometry --
    north_axis: float | None = None
    scale_x: float | None = None
    scale_y: float | None = None
    scale_z: float | None = None


# ---------------------------------------------------------------------------
# Individual mutation helpers
# ---------------------------------------------------------------------------


def _scale_envelope_conductivity(epjson: dict[str, Any], scale: float) -> None:
    """Scale the ``conductivity`` field of every ``Material`` object."""
    for _name, mat in epjson.get("Material", {}).items():
        if "conductivity" in mat:
            mat["conductivity"] = mat["conductivity"] * scale


def _set_window_properties(
    epjson: dict[str, Any],
    u_factor: float | None,
    shgc: float | None,
) -> None:
    """Override U-factor and/or SHGC on all SimpleGlazingSystem windows."""
    for _name, win in epjson.get("WindowMaterial:SimpleGlazingSystem", {}).items():
        if u_factor is not None:
            win["u_factor"] = u_factor
        if shgc is not None:
            win["solar_heat_gain_coefficient"] = shgc


def _scale_infiltration(epjson: dict[str, Any], scale: float) -> None:
    """Scale ``design_flow_rate`` on every ZoneInfiltration:DesignFlowRate."""
    for _name, infil in epjson.get("ZoneInfiltration:DesignFlowRate", {}).items():
        if "design_flow_rate" in infil:
            infil["design_flow_rate"] = infil["design_flow_rate"] * scale


def _set_north_axis(epjson: dict[str, Any], degrees: float) -> None:
    """Set the building rotation via the ``north_axis`` field."""
    for _name, bldg in epjson.get("Building", {}).items():
        bldg["north_axis"] = degrees


def _compute_surface_area(vertices: list[dict[str, float]]) -> float:
    """Compute the area of a planar polygon from its EnergyPlus vertex list
    using the Newell method."""
    n = len(vertices)
    if n < 3:
        return 0.0
    nx = ny = nz = 0.0
    for i in range(n):
        j = (i + 1) % n
        vi = vertices[i]
        vj = vertices[j]
        x_i = vi["vertex_x_coordinate"]
        y_i = vi["vertex_y_coordinate"]
        z_i = vi["vertex_z_coordinate"]
        x_j = vj["vertex_x_coordinate"]
        y_j = vj["vertex_y_coordinate"]
        z_j = vj["vertex_z_coordinate"]
        nx += (y_i - y_j) * (z_i + z_j)
        ny += (z_i - z_j) * (x_i + x_j)
        nz += (x_i - x_j) * (y_i + y_j)
    return 0.5 * math.sqrt(nx * nx + ny * ny + nz * nz)


def _centroid(vertices: list[dict[str, float]]) -> tuple[float, float, float]:
    n = len(vertices)
    cx = sum(v["vertex_x_coordinate"] for v in vertices) / n
    cy = sum(v["vertex_y_coordinate"] for v in vertices) / n
    cz = sum(v["vertex_z_coordinate"] for v in vertices) / n
    return cx, cy, cz


def _set_fenestration_to_wall_ratio(
    epjson: dict[str, Any], target_ratio: float
) -> None:
    """Resize fenestration surfaces to achieve the target window-to-wall ratio.

    For each window, compute its parent wall area and the desired window area,
    then uniformly scale the window vertices around the window centroid.
    """
    surfaces = epjson.get("BuildingSurface:Detailed", {})
    fenestrations = epjson.get("FenestrationSurface:Detailed", {})

    for _fen_name, fen in fenestrations.items():
        if fen.get("surface_type", "").lower() not in ("window", "glazeddoor"):
            continue

        parent_name = fen.get("building_surface_name", "")
        parent = surfaces.get(parent_name)
        if parent is None:
            continue

        parent_area = _compute_surface_area(parent.get("vertices", []))
        if parent_area <= 0:
            continue

        fen_verts = _get_fenestration_vertices(fen)
        if not fen_verts:
            continue

        current_area = _compute_surface_area(fen_verts)
        if current_area <= 0:
            continue

        desired_area = parent_area * target_ratio
        linear_scale = math.sqrt(desired_area / current_area)

        cx, cy, cz = _centroid(fen_verts)
        for v in fen_verts:
            v["vertex_x_coordinate"] = (
                cx + (v["vertex_x_coordinate"] - cx) * linear_scale
            )
            v["vertex_y_coordinate"] = (
                cy + (v["vertex_y_coordinate"] - cy) * linear_scale
            )
            v["vertex_z_coordinate"] = (
                cz + (v["vertex_z_coordinate"] - cz) * linear_scale
            )

        _set_fenestration_vertices(fen, fen_verts)


def _get_fenestration_vertices(
    fen: dict[str, Any],
) -> list[dict[str, float]]:
    """Extract vertices from a FenestrationSurface:Detailed, handling both the
    ``vertices`` list format and the ``vertex_N_*`` field format."""
    if "vertices" in fen:
        return fen["vertices"]

    verts: list[dict[str, float]] = []
    i = 1
    while f"vertex_{i}_x_coordinate" in fen:
        verts.append(
            {
                "vertex_x_coordinate": fen[f"vertex_{i}_x_coordinate"],
                "vertex_y_coordinate": fen[f"vertex_{i}_y_coordinate"],
                "vertex_z_coordinate": fen[f"vertex_{i}_z_coordinate"],
            }
        )
        i += 1
    return verts


def _set_fenestration_vertices(
    fen: dict[str, Any],
    verts: list[dict[str, float]],
) -> None:
    """Write vertices back, matching whichever format the fenestration uses."""
    if "vertices" in fen:
        fen["vertices"] = verts
        return

    for i, v in enumerate(verts, 1):
        fen[f"vertex_{i}_x_coordinate"] = v["vertex_x_coordinate"]
        fen[f"vertex_{i}_y_coordinate"] = v["vertex_y_coordinate"]
        fen[f"vertex_{i}_z_coordinate"] = v["vertex_z_coordinate"]


def _scale_geometry(
    epjson: dict[str, Any],
    sx: float,
    sy: float,
    sz: float,
) -> None:
    """Scale all surface vertex coordinates around the origin.

    Applies to BuildingSurface:Detailed, FenestrationSurface:Detailed, and
    shading surfaces.
    """
    surface_types = [
        "BuildingSurface:Detailed",
        "FenestrationSurface:Detailed",
        "Shading:Building:Detailed",
        "Shading:Zone:Detailed",
        "Shading:Site:Detailed",
    ]
    for stype in surface_types:
        for _name, surf in epjson.get(stype, {}).items():
            if "vertices" in surf:
                for v in surf["vertices"]:
                    v["vertex_x_coordinate"] *= sx
                    v["vertex_y_coordinate"] *= sy
                    v["vertex_z_coordinate"] *= sz
            else:
                i = 1
                while f"vertex_{i}_x_coordinate" in surf:
                    surf[f"vertex_{i}_x_coordinate"] *= sx
                    surf[f"vertex_{i}_y_coordinate"] *= sy
                    surf[f"vertex_{i}_z_coordinate"] *= sz
                    i += 1

    for _name, internal_mass in epjson.get("InternalMass", {}).items():
        if "surface_area" in internal_mass:
            internal_mass["surface_area"] *= sx * sy

    for _name, foundation in epjson.get(
        "SurfaceProperty:ExposedFoundationPerimeter", {}
    ).items():
        if "total_exposed_perimeter" in foundation:
            avg_horizontal_scale = (sx + sy) / 2.0
            foundation["total_exposed_perimeter"] *= avg_horizontal_scale


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def apply_modifications(
    epjson: dict[str, Any], mod: BuildingModification
) -> dict[str, Any]:
    """Apply a ``BuildingModification`` to an epJSON dict (in place) and return it."""

    if mod.envelope_conductivity_scale is not None:
        _scale_envelope_conductivity(epjson, mod.envelope_conductivity_scale)

    if mod.window_u_factor is not None or mod.window_shgc is not None:
        _set_window_properties(epjson, mod.window_u_factor, mod.window_shgc)

    if mod.infiltration_scale is not None:
        _scale_infiltration(epjson, mod.infiltration_scale)

    if mod.north_axis is not None:
        _set_north_axis(epjson, mod.north_axis)

    if mod.fenestration_to_wall_ratio is not None:
        _set_fenestration_to_wall_ratio(epjson, mod.fenestration_to_wall_ratio)

    sx = mod.scale_x if mod.scale_x is not None else 1.0
    sy = mod.scale_y if mod.scale_y is not None else 1.0
    sz = mod.scale_z if mod.scale_z is not None else 1.0
    if (sx, sy, sz) != (1.0, 1.0, 1.0):
        _scale_geometry(epjson, sx, sy, sz)

    return epjson


def convert_to_epjson(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Derivation:
    """Upgrade an IDF and convert it to epJSON -- nothing else.

    This is the intended entry point for the generator: it produces a raw epJSON
    that ``modify_building`` can operate on *before* the rest of the pipeline
    (meters, timestep, controllability) is applied.
    """
    current = upgrade(input_file, energyplus_path, src_version)
    return convert_idf(current, energyplus_path)


@derivation("modified.epjson")
def modify_building(input_epjson: Path, mod: BuildingModification) -> None:
    """Store-integrated derivation: read *input_epjson*, apply *mod*, write result."""
    dst = OUTPUT.get()
    with open(input_epjson, "r") as f:
        epjson = json.load(f)
    apply_modifications(epjson, mod)
    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)
