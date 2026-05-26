import json
from pathlib import Path

from building2building.store import OUTPUT, derivation


@derivation("linked.epjson")
def link_in_schedule(epjson_file: Path, csv_file: Path):
    dst = OUTPUT.get()
    with open(epjson_file, "r") as f:
        epjson = json.load(f)

    for _k, v in epjson["Schedule:File"].items():
        v["file_name"] = str(csv_file)

    with open(dst, "w") as f:
        json.dump(epjson, f, indent=4)
