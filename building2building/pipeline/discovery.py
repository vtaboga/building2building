import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

from cattrs import structure, unstructure

from building2building.pipeline.parse_reports import (
    get_net_conditioned_area,
    get_warmup_days,
)
from building2building.pipeline.steps.outputs import add_all_outputs
from building2building.store import (
    OUTPUT,
    Expression,
    Realizable,
    derivation,
    expression,
)


@dataclass(frozen=True)
class Metadata:
    """Discovery metadata extracted from EnergyPlus simulation."""

    net_conditioned_area: float
    warmup_phases: int
    warmup_days: int


def extract_discovery_metadata(
    epjson: Realizable,
    epw: Realizable,
) -> Expression[Metadata]:
    """
    Prepare epJSON for discovery simulation and extract all metadata.

    This function:
    1. Adds all necessary output configurations (meters, EDD, tabular, SQLite)
    2. Runs a discovery simulation with warmup phase detection
    3. Extracts metadata from simulation outputs

    Metadata extracted:
    - net_conditioned_area: Building floor area in m²
    - warmup_phases: Number of warmup phases (for minergym)
    - warmup_days: Number of warmup days EnergyPlus used

    Args:
        epjson: Path to the building epJSON file
        epw: Path to the weather file

    Returns:
        Expression that resolves to Metadata instance

    Note:
        Uses pyenergyplus.api which requires setup_energyplus_path() to be called first.
    """

    # Add all output configurations needed for discovery simulation
    epjson_with_outputs = add_all_outputs(epjson)

    @derivation("discovery-metadata.json")
    def run_and_extract(epjson: Path, epw: Path):
        """Run simulation and write metadata JSON."""
        import pyenergyplus.api

        out = OUTPUT.get()

        warmup_count = 0

        def warmup_callback(state):
            nonlocal warmup_count
            warmup_count += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            api = pyenergyplus.api.EnergyPlusAPI()
            state = api.state_manager.new_state()

            api.runtime.callback_after_new_environment_warmup_complete(
                state, warmup_callback
            )

            api.runtime.run_energyplus(
                state,
                [
                    "-d",
                    str(tmpdir),
                    "-w",
                    str(epw),
                    str(epjson),
                ],
            )

            # Extract metadata from simulation outputs
            htm_file = Path(tmpdir) / "eplustbl.htm"
            area = get_net_conditioned_area(htm_file)
            warmup_days = get_warmup_days(htm_file)

        # Create Metadata instance and serialize with cattrs
        metadata = Metadata(
            net_conditioned_area=area,
            warmup_phases=warmup_count,
            warmup_days=warmup_days,
        )

        with open(out, "w") as f:
            json.dump(unstructure(metadata), f, indent=2)

    @expression()
    def parse_metadata(json_path: Path) -> Metadata:
        """Parse metadata JSON into Metadata dataclass."""
        with open(json_path, "r") as f:
            metadata_dict = json.load(f)

        return structure(metadata_dict, Metadata)

    return parse_metadata(run_and_extract(epjson_with_outputs, epw))
