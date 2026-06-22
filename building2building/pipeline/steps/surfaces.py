import json
import logging
from pathlib import Path

from building2building.store import Derivation, OUTPUT, derivation

logger = logging.getLogger(__name__)


@derivation("glued")
def GlueSurfaces(
    input: Path,
):
    dst = OUTPUT.get()
    with open(input, "r") as f:
        epjson = json.load(f)

    # This is a simplified version - would need minergym.ontology for full implementation
    surfaces = epjson.get("BuildingSurface:Detailed", {})

    # Basic surface matching by coordinates (simplified)
    # In the full version, this would use the ontology to find overlapping surfaces
    logger.info(f"Processing {len(surfaces)} surfaces for gluing")

    # For now, just copy the input to output
    # The full implementation would require the ontology library
    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)


def glue_surfaces(epjson_in: Derivation) -> Derivation:
    """Glue together overlapping surfaces."""
    return GlueSurfaces(epjson_in)
