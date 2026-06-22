"""Per-zone geometric attributes derived from a building's epJSON.

Each conditioned zone gets a fixed-arity tuple of normalized scalars
(centroid + bbox size, normalized to the building bbox; floor / exterior
wall / ground-contact area fractions). These values are intrinsic to
the building (computable from `BuildingSurface:Detailed` alone) and
land as :attr:`MorphologyNode.attributes` for zone-typed nodes — see
:func:`building2building.morphology.build_morphology`.

Reads `BuildingSurface:Detailed` via :class:`minergym.ontology.Ontology`,
the same EnergyPlus traversal layer used elsewhere in the pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from minergym.ontology import Ontology

# Names for the per-zone attribute slots, in the order that
# `ZoneGeometry.to_array` emits them. The corresponding bounds live on
# the zone NodeTypes (`_attr_low` / `_attr_high`).
ZONE_ATTRIBUTE_NAMES: tuple[str, ...] = (
    "centroid_x",
    "centroid_y",
    "centroid_z",
    "size_x",
    "size_y",
    "size_z",
    "floor_area_frac",
    "ext_wall_area_frac",
    "ground_contact_frac",
)
ZONE_ATTRIBUTE_DIM: int = len(ZONE_ATTRIBUTE_NAMES)
assert ZONE_ATTRIBUTE_DIM == 9


# Names of the :class:`NodeType`s in :data:`ALL_NODE_TYPES` that
# represent thermal zones (and therefore carry geometry attributes).
ZONE_TYPE_NAMES: frozenset[str] = frozenset(
    {
        "unitary_zone",
        "vav_zone",
        "vav_zone_no_cooling",
        "heating_zone",
        "uncontrolled_zone",
    }
)


@dataclass(frozen=True)
class ZoneGeometry:
    """Typed view of one zone's geometric attributes.

    Used at the epJSON-extraction boundary and at the b2b ↔ downstream
    boundary (e.g. morel/RL adapters); on :class:`MorphologyNode` the
    values are stored as a flat ndarray in :data:`ZONE_ATTRIBUTE_NAMES`
    order via :meth:`to_array`.

    All values are dimensionless, in `[0, 1]`:
      * `centroid`, `size`: fractions of the building's bounding box.
      * `floor_area_frac`, `ext_wall_area_frac`: zone floor / exterior
        wall area divided by the building-wide totals.
      * `ground_contact_frac`: fraction of this zone's floor area whose
        EnergyPlus outside-boundary-condition is ground-contacting.
    """

    centroid: tuple[float, float, float]
    size: tuple[float, float, float]
    floor_area_frac: float
    ext_wall_area_frac: float
    ground_contact_frac: float

    def to_array(self) -> np.ndarray:
        """Flat float32 array in :data:`ZONE_ATTRIBUTE_NAMES` order."""
        return np.array(
            (
                *self.centroid,
                *self.size,
                self.floor_area_frac,
                self.ext_wall_area_frac,
                self.ground_contact_frac,
            ),
            dtype=np.float32,
        )


def _polygon_area_3d(points: list[tuple[float, float, float]]) -> float:
    """Area of an arbitrary planar polygon in 3D, via cross-product sum."""
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    n = np.zeros(3)
    for i in range(len(pts)):
        n += np.cross(pts[i], pts[(i + 1) % len(pts)])
    return 0.5 * float(np.linalg.norm(n))


def _surface_attr(ont: Ontology, surface, predicate: str) -> str | None:
    """Single-valued lookup of `surface idf:<predicate> ?o` via SPARQL."""
    q = f"""# -*- mode: sparql -*-
    SELECT ?val
    WHERE {{ ?surface idf:{predicate} ?val . }}"""
    rows = list(ont.rdf.query(q, initBindings={"surface": surface}))
    if not rows:
        return None
    return rows[0].val.toPython()


def extract_zone_geometry(
    epjson: dict[str, Any] | Path | str,
) -> dict[str, ZoneGeometry]:
    """Compute per-zone :class:`ZoneGeometry` keyed by zone name.

    Args:
        epjson: Parsed epJSON dict, or a path to a `building.epjson` file.

    Returns:
        Mapping ``zone_name -> ZoneGeometry``. Zones with no surfaces
        (shouldn't happen for valid EnergyPlus inputs) are omitted.
    """
    if isinstance(epjson, (str, Path)):
        with open(epjson) as f:
            epjson = json.load(f)
    ont = Ontology.from_object(epjson)
    hierarchy = ont.zone_surface_point_hierarchy()

    all_pts = np.array(
        [v for surfs in hierarchy.values() for verts in surfs.values() for v in verts],
        dtype=float,
    )
    if all_pts.size == 0:
        return {}
    b_min, b_max = all_pts.min(axis=0), all_pts.max(axis=0)
    b_size = np.where((b_max - b_min) > 0, b_max - b_min, 1.0)

    total_floor_area = 0.0
    total_ext_wall_area = 0.0
    surface_meta: dict[Any, tuple[str | None, str | None, float]] = {}
    for surfs in hierarchy.values():
        for surface, verts in surfs.items():
            stype = _surface_attr(ont, surface, "surface_type")
            obc = _surface_attr(ont, surface, "outside_boundary_condition")
            area = _polygon_area_3d(verts)
            surface_meta[surface] = (stype, obc, area)
            if stype == "Floor":
                total_floor_area += area
            if stype == "Wall" and obc == "Outdoors":
                total_ext_wall_area += area

    out: dict[str, ZoneGeometry] = {}
    for zone, surfs in hierarchy.items():
        zone_name = str(zone)
        zone_pts = np.array(
            [v for verts in surfs.values() for v in verts],
            dtype=float,
        )
        if zone_pts.size == 0:
            continue
        z_min, z_max = zone_pts.min(axis=0), zone_pts.max(axis=0)
        centroid = ((z_min + z_max) / 2 - b_min) / b_size
        size = (z_max - z_min) / b_size

        floor_area = 0.0
        ext_wall_area = 0.0
        ground_floor_area = 0.0
        for surface in surfs:
            stype, obc, area = surface_meta[surface]
            if stype == "Floor":
                floor_area += area
                # EnergyPlus encodes ground-contacting floors with several
                # BC variants (Ground, GroundFCfactorMethod, Foundation,
                # ...). Anything starting with "Ground" plus Foundation
                # counts.
                if obc and (obc.startswith("Ground") or obc == "Foundation"):
                    ground_floor_area += area
            elif stype == "Wall" and obc == "Outdoors":
                ext_wall_area += area

        out[zone_name] = ZoneGeometry(
            centroid=tuple(float(x) for x in centroid),
            size=tuple(float(x) for x in size),
            floor_area_frac=(
                float(floor_area / total_floor_area) if total_floor_area > 0 else 0.0
            ),
            ext_wall_area_frac=(
                float(ext_wall_area / total_ext_wall_area)
                if total_ext_wall_area > 0
                else 0.0
            ),
            ground_contact_frac=(
                float(ground_floor_area / floor_area) if floor_area > 0 else 0.0
            ),
        )
    return out
