import simulator.simulation as simulation
from simulator.data.building import crawlspace
from simulator.data.weather import honolulu
from importlib import resources
import pytest


def test_simulation_separate() -> None:
    sim1 = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, None, {}, max_steps=100
    )
    sim2 = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, None, {}, max_steps=100
    )

    assert sim1.obs_chan is not sim2.obs_chan


def test_simulation_runs() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, None, {}, max_steps=100)

    obs, done = sim.start()
    while not done:
        assert obs is None
        obs, done = sim.step({})


def test_simulation_obs() -> None:

    obs_template = {
        "temp": simulation.VariableHole("ZONE AIR TEMPERATURE", "crawlspace")
    }

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, obs_template, {}, max_steps=100
    )
    obs, done = sim.start()
    assert obs["temp"] != 0.0


def test_simulation_invalid_variable() -> None:

    # an invalid variable
    obs_template = simulation.VariableHole("", "")

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, obs_template, {}, max_steps=100
    )

    with pytest.raises(simulation.InvalidVariable) as exn_info:
        obs, done = sim.start()


def test_simulation_invalid_meter() -> None:

    obs_template = simulation.MeterHole("")

    sim = simulation.EnergyPlusSimulation(
        crawlspace, honolulu, obs_template, {}, max_steps=100
    )

    with pytest.raises(simulation.InvalidMeter) as exn_info:
        obs, done = sim.start()


def test_simulation_time() -> None:
    obs = {}
    obs["current_time"] = simulation.FunctionHole(simulation.api.exchange.current_time)
    obs["day_of_year"] = simulation.FunctionHole(simulation.api.exchange.day_of_year)

    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, obs, {}, max_steps=1000)
    obs, done = sim.start()
    while True:
        new_obs, done = sim.step({})
        if done:
            break
        # We would expect the day of year to go up linearly. However, this is
        # not currently (2024-11-11T12:34:20) the case. It seems that after the
        # initial warmup is done, a single day will pass (day 202) before a
        # second period of warmup which will be followed by the start of the
        # real RunPeriod (day 1).
        assert new_obs["day_of_year"] - obs["day_of_year"] in [0, 1]
        obs = new_obs


def test_simulation_missing_file() -> None:
    sim = simulation.EnergyPlusSimulation("does_not_exist", honolulu, {}, {})

    with pytest.raises(simulation.SimulationCrashed):
        sim.start()


def test_simulation_repr() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, {}, {})
    print(sim)


def test_get_api_endpoints() -> None:
    sim = simulation.EnergyPlusSimulation(crawlspace, honolulu, {}, {})
    sim.start()

    # Will crash is some unknown api object is encountered
    sim.get_api_endpoints()
