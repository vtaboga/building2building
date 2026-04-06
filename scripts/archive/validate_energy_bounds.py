"""Parse existing EnergyPlus HTML reports to validate energy observation bounds.

Extracts peak HVAC demand (W) and conditioned area (m²) from eplustbl.htm files,
then computes peak Wh/m² per 15-minute timestep to verify the normalization interval.

Uses regex for fast extraction (avoids full HTML parsing of multi-MB files).
"""

import re
import sys
from pathlib import Path

_AREA_RE = re.compile(
    r"Net Conditioned Building Area.*?<td[^>]*>\s*([\d.]+)\s*</td>",
    re.DOTALL,
)

_PEAK_TABLE_RE = re.compile(
    r"(End Uses\b.*?Electricity \[W\].*?</table>)",
    re.DOTALL,
)

_ROW_RE = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_CELL_RE = re.compile(r"<td[^>]*>(.*?)</td>", re.DOTALL)


def parse_float(s: str) -> float | None:
    s = s.strip()
    if not s or s == "-":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def extract_peak_hvac_and_area(htm_path: Path) -> dict | None:
    text = htm_path.read_text(encoding="utf-8", errors="replace")

    # Area
    m = _AREA_RE.search(text)
    if m is None:
        return None
    area = parse_float(m.group(1))
    if area is None or area <= 0:
        return None

    # Find the peak demand table (End Uses with [W] columns)
    peak_match = _PEAK_TABLE_RE.search(text)
    if peak_match is None:
        return None

    table_html = peak_match.group(1)
    peak_hvac_elec_w = 0.0
    peak_hvac_gas_w = 0.0

    for row_m in _ROW_RE.finditer(table_html):
        cells = [c.strip() for c in _CELL_RE.findall(row_m.group(1))]
        if not cells:
            continue
        label = cells[0].strip()
        if label in ("Heating", "Cooling", "Fans"):
            if len(cells) > 1:
                v = parse_float(cells[1])
                if v is not None:
                    peak_hvac_elec_w += v
            if len(cells) > 2:
                v = parse_float(cells[2])
                if v is not None:
                    peak_hvac_gas_w += v

    total_peak_w = peak_hvac_elec_w + peak_hvac_gas_w
    peak_wh_per_m2_15min = total_peak_w * 0.25 / area

    return {
        "area_m2": area,
        "peak_hvac_elec_w": peak_hvac_elec_w,
        "peak_hvac_gas_w": peak_hvac_gas_w,
        "total_peak_hvac_w": total_peak_w,
        "peak_wh_per_m2_15min": peak_wh_per_m2_15min,
    }


def main() -> None:
    outputs_dir = Path(__file__).resolve().parent.parent / "outputs"
    fixtures_dir = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

    htm_files = sorted(outputs_dir.rglob("eplustbl.htm"))
    fixture_htm = fixtures_dir / "eplustbl.htm"
    if fixture_htm.exists():
        htm_files.append(fixture_htm)

    if not htm_files:
        print("No eplustbl.htm files found.")
        sys.exit(1)

    print(f"Parsing {len(htm_files)} files...")

    results_by_type: dict[str, list[dict]] = {}
    btypes = [
        "Warehouse", "RetailStandalone",
        "RestaurantFastFood", "OfficeMedium", "OfficeSmall",
    ]

    for i, htm in enumerate(htm_files):
        parts = str(htm)
        btype = "Unknown"
        for bt in btypes:
            if bt in parts:
                btype = bt
                break

        result = extract_peak_hvac_and_area(htm)
        if result is None:
            continue
        result["path"] = str(htm)
        results_by_type.setdefault(btype, []).append(result)

        if (i + 1) % 50 == 0:
            print(f"  ... parsed {i + 1}/{len(htm_files)}")

    print()
    print("=" * 80)
    print("ENERGY BOUNDS VALIDATION: Peak HVAC energy per 15-min timestep (Wh/m²)")
    print("=" * 80)

    global_max = 0.0
    for btype in sorted(results_by_type.keys()):
        records = results_by_type[btype]
        peaks = [r["peak_wh_per_m2_15min"] for r in records]
        areas = [r["area_m2"] for r in records]
        max_peak = max(peaks)
        global_max = max(global_max, max_peak)

        print(f"\n{btype} ({len(records)} reports):")
        print(f"  Area range:             {min(areas):.1f} - {max(areas):.1f} m²")
        print(f"  Peak Wh/m²/15min range: {min(peaks):.2f} - {max(peaks):.2f}")
        print(f"  Max peak Wh/m²/15min:   {max_peak:.2f}")

        worst = max(records, key=lambda r: r["peak_wh_per_m2_15min"])
        print(f"  Worst case: area={worst['area_m2']:.1f}m², "
              f"elec={worst['peak_hvac_elec_w']:.0f}W, "
              f"gas={worst['peak_hvac_gas_w']:.0f}W")

    print(f"\n{'=' * 80}")
    print(f"GLOBAL MAX peak Wh/m²/15min: {global_max:.2f}")
    if global_max <= 50:
        print("Current interval [0, 50] is SUFFICIENT")
    else:
        suggested = int(global_max * 1.2) + 1
        print(f"Current interval [0, 50] INSUFFICIENT — suggest [0, {suggested}]")
    print("=" * 80)


if __name__ == "__main__":
    main()
