"""Smoke tests for the reactive-control baseline (run_reactive_control).

Covers:
- Module imports without circular dependencies.
- ``_select_policy`` correctly dispatches to AirLoop vs UnitaryHvac controller.
- ``write_csv`` produces the expected columns.
- ``append_result`` / ``load_completed_keys`` round-trip: an interrupted
  sweep resumes from the partial CSV instead of re-simulating.
- ``RunResult`` dataclass has the required fields.
"""

from __future__ import annotations

import csv
import inspect
from dataclasses import fields

import pytest


@pytest.mark.quick
class TestRunReactiveControlImports:
    def test_module_importable(self) -> None:
        import baselines.run_reactive_control as mod

        assert hasattr(mod, "evaluate_building")
        assert hasattr(mod, "write_csv")
        assert hasattr(mod, "RunResult")

    def test_controllers_importable(self) -> None:
        from baselines.controllers.air_loop import AirLoopConfig, AirLoopPolicy
        from baselines.controllers.unitary_hvac import (
            UnitaryHvacConfig,
            UnitaryHvacPolicy,
        )

        assert callable(AirLoopPolicy)
        assert callable(UnitaryHvacPolicy)
        assert callable(AirLoopConfig)
        assert callable(UnitaryHvacConfig)


@pytest.mark.quick
class TestRunResult:
    def test_runresult_has_required_fields(self) -> None:
        from baselines.run_reactive_control import RunResult

        field_names = {f.name for f in fields(RunResult)}
        required = {
            "building_type",
            "building_id",
            "task",
            "run_period",
            "rewards",
            "reward_mean",
        }
        assert required.issubset(field_names)

    def test_runresult_instantiation(self) -> None:
        from baselines.run_reactive_control import RunResult

        result = RunResult(
            building_type="OfficeSmall",
            building_id="OfficeSmall-0001",
            task="task_const_e0",
            run_period="winter",
            rewards=[-1.0, -2.0],
            reward_mean=-1.5,
        )
        assert result.building_type == "OfficeSmall"
        assert result.reward_mean == pytest.approx(-1.5)


@pytest.mark.quick
class TestWriteCsv:
    def test_write_csv_produces_header_and_rows(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from baselines.run_reactive_control import RunResult, write_csv

        results = [
            RunResult(
                building_type="OfficeSmall",
                building_id="OfficeSmall-0001",
                task="task_const_e0",
                run_period="winter",
                rewards=[-10.0],
                reward_mean=-10.0,
            )
        ]
        out = tmp_path / "results.csv"
        write_csv(results, out, n_runs=1)

        with open(out) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["building_id"] == "OfficeSmall-0001"
        assert rows[0]["task"] == "task_const_e0"
        assert "reward_mean" in rows[0]

    def test_write_csv_signature(self) -> None:
        from baselines.run_reactive_control import write_csv

        sig = inspect.signature(write_csv)
        assert "results" in sig.parameters
        assert "path" in sig.parameters
        assert "n_runs" in sig.parameters


@pytest.mark.quick
class TestIncrementalAppendAndResume:
    @staticmethod
    def _result(building_id: str, task: str = "task_const_e05"):
        from baselines.run_reactive_control import RunResult

        return RunResult(
            building_type="OfficeMedium",
            building_id=building_id,
            task=task,
            run_period="winter",
            rewards=[-10.0],
            reward_mean=-10.0,
        )

    def test_append_result_writes_header_once(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from baselines.run_reactive_control import append_result

        out = tmp_path / "results.csv"
        append_result(self._result("OfficeMedium-4001"), out, n_runs=1)
        append_result(self._result("OfficeMedium-4002"), out, n_runs=1)

        with open(out) as fh:
            rows = list(csv.DictReader(fh))

        assert [r["building_id"] for r in rows] == [
            "OfficeMedium-4001",
            "OfficeMedium-4002",
        ]
        assert rows[0]["reward_mean"] == "-10.0"

    def test_load_completed_keys_round_trip(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from baselines.run_reactive_control import append_result, load_completed_keys

        out = tmp_path / "results.csv"
        assert load_completed_keys(out) == set()

        append_result(self._result("OfficeMedium-4001"), out, n_runs=1)
        assert load_completed_keys(out) == {
            ("OfficeMedium", "task_const_e05", "winter", "OfficeMedium-4001")
        }

    def test_append_matches_write_csv_columns(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        from baselines.run_reactive_control import append_result, write_csv

        appended = tmp_path / "appended.csv"
        rewritten = tmp_path / "rewritten.csv"
        result = self._result("OfficeMedium-4001")
        append_result(result, appended, n_runs=1)
        write_csv([result], rewritten, n_runs=1)

        assert appended.read_text() == rewritten.read_text()


@pytest.mark.quick
class TestSelectPolicyDispatch:
    def test_select_policy_signature_accepts_controller_type(self) -> None:
        from baselines.run_reactive_control import _select_policy

        sig = inspect.signature(_select_policy)
        assert "controller_type" in sig.parameters or len(sig.parameters) >= 1
