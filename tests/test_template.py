import simulator.template as template
from dataclasses import dataclass


@dataclass
class Thing:
    first_field: int
    second_field: str


def none(_):
    return None


def test_search_replace_dict():
    t1 = {"a": Thing(1, "a"), "b": Thing(2, "b")}

    t2 = {"a": None, "b": None}

    assert template.search_replace(t1, Thing, none) == t2


def test_search_replace_list():
    t1 = [Thing(1, 2), Thing(2, 3)]
    t2 = [None, None]

    assert template.search_replace(t1, Thing, none) == t2


def test_search_replace_tuple():
    t1 = (Thing(1, "a"), Thing(2, "b"))

    t2 = (None, None)

    assert template.search_replace(t1, Thing, none) == t2


def test_search_replace_other():
    t1 = ["there can be anything here", (1, 2, {"x": None})]

    assert template.search_replace(t1, Thing, none) == t1
