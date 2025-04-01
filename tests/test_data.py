import simulator.data.building as building
import simulator.data.weather as weather
import os


def test_crawlspace():
    assert os.path.exists(building.crawlspace)


def test_honolulu():
    assert os.path.exists(weather.honolulu)


def test_all_weather_files():
    for f in weather.all_weather_files:
        assert os.path.exists(f)


def test_all_building_files():
    for f in building.all_building_files:
        assert os.path.exists(f)
