from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

from building2building.pipeline import infer_actuator_bounds_from_sql


def test_infer_actuator_bounds_from_sql_uses_max_heating_and_cooling() -> None:
    with tempfile.TemporaryDirectory() as td:
        sql_path = Path(td) / "eplusout.sql"
        con = sqlite3.connect(str(sql_path))
        cur = con.cursor()

        # Minimal schema subset used by infer_actuator_bounds_from_sql
        cur.execute(
            """
            CREATE TABLE ReportDataDictionary (
                ReportDataDictionaryIndex INTEGER PRIMARY KEY,
                KeyValue TEXT,
                Name TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE ReportData (
                ReportDataDictionaryIndex INTEGER,
                Value REAL
            )
            """
        )

        # Two keys (two unitary systems) with different maxima
        d = [
            (1, "UNIT_A", "Unitary System Sensible Heating Rate"),
            (2, "UNIT_A", "Unitary System Sensible Cooling Rate"),
            (3, "UNIT_B", "Unitary System Sensible Heating Rate"),
            (4, "UNIT_B", "Unitary System Sensible Cooling Rate"),
        ]
        cur.executemany(
            "INSERT INTO ReportDataDictionary(ReportDataDictionaryIndex, KeyValue, Name) VALUES (?,?,?)",
            d,
        )

        # UNIT_A: max heat=10, max cool=7
        cur.executemany(
            "INSERT INTO ReportData(ReportDataDictionaryIndex, Value) VALUES (?,?)",
            [(1, 1.0), (1, 10.0), (2, 3.0), (2, 7.0)],
        )
        # UNIT_B: max heat=0 (no heating), max cool=5
        cur.executemany(
            "INSERT INTO ReportData(ReportDataDictionaryIndex, Value) VALUES (?,?)",
            [(3, 0.0), (4, 5.0)],
        )

        con.commit()
        con.close()

        bounds = infer_actuator_bounds_from_sql(sql_path=sql_path)

        assert bounds["Unitary HVAC::Sensible Load Request::UNIT_A"] == {
            "lower_bound": -7.0,
            "upper_bound": 10.0,
        }
        assert bounds["Unitary HVAC::Sensible Load Request::UNIT_B"] == {
            "lower_bound": -5.0,
            "upper_bound": 0.0,
        }


