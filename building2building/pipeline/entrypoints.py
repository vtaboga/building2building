from building2building.pipeline.pipelines import create_discovery_pipeline, create_control_pipeline
from building2building.store import Derivation, Realizable


def create_complete_pipeline(
    input_file: Derivation,
    energyplus_path: Realizable,
    src_version: str,
) -> Derivation:
    """Create a complete processing pipeline from raw IDF to ready-to-go epJSON."""
    discovery = create_discovery_pipeline(input_file, energyplus_path, src_version)
    return create_control_pipeline(discovery)

