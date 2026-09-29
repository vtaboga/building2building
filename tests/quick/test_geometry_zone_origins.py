"""Zone origins must be applied before any geometric attribute is computed.

EnergyPlus's default ``GlobalGeometryRules.coordinate_system = Relative``
makes every ``BuildingSurface:Detailed`` vertex zone-relative; the ``Zone``
object's ``x/y/z_origin`` (and ``direction_of_relative_north``) place it
in the building. The DOE stand-alone retail prototype (b2b's
RetailStandalone family) uses this: four of its five zones have a
non-zero origin, so measuring the raw vertices put the stockroom at the
front and stacked the two front zones on top of each other.
"""
from __future__ import annotations

import math

import pytest
from minergym.ontology import Ontology

from building2building.geometry import (
    ZonePlacement,
    extract_zone_geometry,
    zone_placements,
    zone_surface_building_vertices,
)


def _box(zone: str, w: float, d: float, h: float = 3.0) -> dict:
    """Floor + roof + 4 walls of a w x d x h box at the zone's local origin."""
    def surf(name, stype, verts, obc="Outdoors"):
        return {
            f"{zone}_{name}": {
                "surface_type": stype, "zone_name": zone,
                "outside_boundary_condition": obc,
                "vertices": [{"vertex_x_coordinate": x, "vertex_y_coordinate": y,
                              "vertex_z_coordinate": z} for x, y, z in verts],
            }
        }
    out = {}
    out.update(surf("floor", "Floor", [(0, 0, 0), (0, d, 0), (w, d, 0), (w, 0, 0)], "Ground"))
    out.update(surf("roof", "Roof", [(0, 0, h), (w, 0, h), (w, d, h), (0, d, h)]))
    out.update(surf("w1", "Wall", [(0, 0, h), (0, 0, 0), (w, 0, 0), (w, 0, h)]))
    out.update(surf("w2", "Wall", [(w, 0, h), (w, 0, 0), (w, d, 0), (w, d, h)]))
    out.update(surf("w3", "Wall", [(w, d, h), (w, d, 0), (0, d, 0), (0, d, h)]))
    out.update(surf("w4", "Wall", [(0, d, h), (0, d, 0), (0, 0, 0), (0, 0, h)]))
    return out


def _epjson(coordinate_system: str | None, zone_b: dict) -> dict:
    """Two 10 x 10 boxes: A at the origin, B placed by ``zone_b``."""
    ep = {
        "Zone": {"A": {}, "B": zone_b},
        "BuildingSurface:Detailed": {**_box("A", 10, 10), **_box("B", 10, 10)},
    }
    if coordinate_system is not None:
        ep["GlobalGeometryRules"] = {"Rules": {"coordinate_system": coordinate_system}}
    return ep


def test_relative_origin_places_zone_in_building_frame():
    ont = Ontology.from_object(_epjson("Relative", {"x_origin": 10.0, "y_origin": 0.0}))
    pl = zone_placements(ont)
    assert pl["A"] == ZonePlacement()
    assert pl["B"].origin == (10.0, 0.0, 0.0)
    verts = zone_surface_building_vertices(ont)
    xs = {v[0] for surfs in verts.values() for s, vv in surfs.items() if str(s).startswith("B_") for v in vv}
    assert xs == {10.0, 20.0}
    g = extract_zone_geometry(ont)
    # building bbox is now x in [0, 20]: A centred at 0.25, B at 0.75
    assert g["A"].centroid[0] == pytest.approx(0.25)
    assert g["B"].centroid[0] == pytest.approx(0.75)
    assert g["A"].size[0] == pytest.approx(0.5)
    # rigid placement: area-based fractions are untouched
    assert g["A"].floor_area_frac == pytest.approx(0.5)
    assert g["B"].floor_area_frac == pytest.approx(0.5)


def test_missing_rules_default_to_relative():
    ont = Ontology.from_object(_epjson(None, {"x_origin": 10.0}))
    assert extract_zone_geometry(ont)["B"].centroid[0] == pytest.approx(0.75)


@pytest.mark.parametrize("cs", ["Absolute", "World", "absolute"])
def test_absolute_coordinates_ignore_zone_origins(cs):
    ont = Ontology.from_object(_epjson(cs, {"x_origin": 10.0}))
    assert zone_placements(ont)["B"] == ZonePlacement()
    g = extract_zone_geometry(ont)
    assert g["A"].centroid[0] == pytest.approx(g["B"].centroid[0])


def test_relative_north_rotates_clockwise_about_zone_origin():
    pl = ZonePlacement(origin=(5.0, 0.0, 0.0), relative_north_deg=90.0)
    # local +y (north) becomes building +x after a 90 deg clockwise turn
    x, y, z = pl.to_building((0.0, 1.0, 0.0))
    assert (x, y, z) == pytest.approx((6.0, 0.0, 0.0), abs=1e-12)
    x, y, z = pl.to_building((1.0, 0.0, 0.0))
    assert (x, y, z) == pytest.approx((5.0, -1.0, 0.0), abs=1e-12)


def test_minimal_retailstandalone_fixture_layout():
    """The committed RetailStandalone fixture (DOE prototype origins): the
    stockroom is at the back, the two front zones sit side by side, and the
    vestibule is between them on the front facade."""
    import json
    from pathlib import Path

    p = Path(__file__).resolve().parents[1] / "fixtures" / "minimal_retailstandalone" / "building.epjson"
    g = extract_zone_geometry(Ontology.from_object(json.loads(p.read_text())))
    assert g["Back_Space"].centroid[1] > 0.8
    assert g["Core_Retail"].centroid[1] == pytest.approx(0.45, abs=0.1)
    front = [g[z].centroid[1] for z in ("Point_Of_Sale", "Front_Entry", "Front_Retail")]
    assert max(front) < 0.15
    assert g["Point_Of_Sale"].centroid[0] < g["Front_Entry"].centroid[0] < g["Front_Retail"].centroid[0]
    assert g["Point_Of_Sale"].centroid != g["Front_Retail"].centroid
