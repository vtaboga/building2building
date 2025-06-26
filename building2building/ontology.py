import rdflib
from rdflib.namespace import RDF
from rdflib.term import Node
from typing import List, TypeAlias, Self, TypeVar, reveal_type
from dataclasses import dataclass
from pathlib import Path
import json

T = TypeVar("T")
Point: TypeAlias = tuple[float, float, float]
ZoneSurfacePointHierarchy: TypeAlias = dict[Node, dict[Node, List[Point]]]
UndirectedGraph: TypeAlias = dict[T, frozenset[T]]

IDF = rdflib.Namespace("https://energyplus.net/")

def create_rdf_list(graph: rdflib.Graph, items: list[Node]):
    last = RDF.nil
    for item in reversed(items):
        new_node = rdflib.BNode()
        # graph.add((new_node, RDF.type, RDF.))
        graph.add((new_node, RDF.first, item))
        graph.add((new_node, RDF.rest, last))
        last = new_node
        graph.add((last, RDF.type, RDF.List))

    return last

def intern_object(graph: rdflib.Graph, obj) -> Node:
    if isinstance(obj, str):
        return rdflib.Literal(obj)
    elif isinstance(obj, float):
        return rdflib.Literal(obj)
    elif isinstance(obj, int):
        return rdflib.Literal(obj)
    elif isinstance(obj, list):
        things = [intern_object(graph, v) for v in obj]
        return create_rdf_list(graph, things)
    elif isinstance(obj, dict):
        node = rdflib.BNode()

        for k, v in obj.items():
            interned_v = intern_object(graph, v)
            graph.add((node, IDF[k], interned_v))

        return node
    else:
        raise Exception(f"huh? {obj}, {type(obj)}")


@dataclass
class Ontology:
    rdf: rdflib.Graph

    @classmethod
    def from_object(cls, obj) -> Self:
        g = rdflib.Graph()

        for type_name, value in obj.items():
            for obj_name, contents in value.items():
                g.add((rdflib.Literal(obj_name), RDF.type, rdflib.Literal(type_name)))

                for attr_name, value in contents.items():
                    interned = intern_object(g, value)
                    g.add((rdflib.Literal(obj_name), IDF[attr_name], interned))

        g.bind("idf", IDF)
        return cls(g)

    @classmethod
    def from_json(cls, path: Path) -> Self:
        """Take a rdf graph, a rdf subject and a python dict/array and intern
        it into the ontology.

        Dictionnaries are interned using the keys as predicates, lists use
        has-elem.
        """

        with open(path, "rb") as f:
            obj = json.load(f)

        return cls.from_object(obj)


    def all_triples(self) -> list[tuple[Node, Node, Node]]:
        """Return every single triple in the ontology."""

        return [
            (a,b,c) for (a,b,c) in self.rdf.query("SELECT ?a ?b ?c WHERE {?a ?b ?c .}")
        ]

    def zones(self) -> List[Node]:
        """Return every node that is a "Zone"."""

        q = """# -*- mode: sparql -*-# -*- mode: sparql -*-
    SELECT ?zone
    WHERE {
      ?zone a "Zone" .
    }"""
        return list(set(r.zone for r in self.rdf.query(q)))

    def surfaces(self):
        """Return every single `BuildingSurface:Detailed` in the ontology."""

        q = """# -*- mode: sparql -*-
    SELECT ?name
    WHERE {
      ?name a "BuildingSurface:Detailed" .
    }"""
        return list(self.rdf.query(q))

    def zone_surfaces(self, zone: Node) -> list[Node]:
        """Return all the surfaces that have `zone_name` equal to `zone`."""

        q = """# -*- mode: sparql -*-
    SELECT ?surface
    WHERE {
      ?surface a "BuildingSurface:Detailed" .
      ?surface idf:zone_name ?zone .
    }"""
        return [r.surface for r in self.rdf.query(q, initBindings={"zone": zone})]


    def surface_vertices(self, surface: Node) -> list[Point]:
        """Return the vertices of a surface."""
        q = """# -*- mode: sparql -*-

    SELECT ?x ?y ?z
    WHERE {
      ?surface idf:vertices ?vertices .
      ?vertices rdf:rest*/rdf:first ?vertex .

      ?vertex idf:vertex_x_coordinate ?x .
      ?vertex idf:vertex_y_coordinate ?y .
      ?vertex idf:vertex_z_coordinate ?z .
    }"""
        return [
            (x.toPython(), y.toPython(), z.toPython())
            for (x,y,z) in self.rdf.query(q, initBindings={"surface": surface})
        ]

    def zone_surface_point_hierarchy(self) -> ZoneSurfacePointHierarchy:
        return {
            z : {
                s : [v for v in self.surface_vertices(s)]
                for s in self.zone_surfaces(z)
            }
            for z in self.zones()
        }

    def pointset_to_surfaceset(self) -> dict[frozenset[Point], set[Node]]:
        """Return a mapping (set of vertices) -> (set of surfaces) useful for
        detecting which surfaces share all of their point and for extracting an
        adjacency relation.

        """

        hierarchy = self.zone_surface_point_hierarchy()

        pointset_to_surfaces = {}

        for zone, surfaces in hierarchy.items():
            for surface, vertices in surfaces.items():
                pointset_to_surfaces.setdefault(frozenset(vertices), set()).add(surface)

        return pointset_to_surfaces


    def zone_adjacency(self) -> UndirectedGraph[Node]:
        """Compute a graph of adjacent zones using `BuildingSurface:Detailed`
        objects and the `outside_boundary_condition_object` property."""

        q = """# -*- mode: sparql -*-
SELECT ?zoneA ?zoneB
WHERE {
  ?surfaceA a "BuildingSurface:Detailed" .
  ?surfaceA idf:zone_name ?zoneA .

  ?surfaceB a "BuildingSurface:Detailed" .
  ?surfaceB idf:zone_name ?zoneB .

  # One is the other side of the other
  ?surfaceA idf:outside_boundary_condition "Surface" .
  ?surfaceA idf:outside_boundary_condition_object ?surfaceB .

  # And vice versa
  ?surfaceB idf:outside_boundary_condition "Surface" .
  ?surfaceB idf:outside_boundary_condition_object ?surfaceA .
}
"""

        out = {}

        for r in self.rdf.query(q):
            out.setdefault(r.zoneA, set()).add(r.zoneB)
            out.setdefault(r.zoneB, set()).add(r.zoneB)

        return out

    def schedules(self) -> List[Node]:
        """Return a list of all the scheduler names in the building"""
        q = """# -*- mode: sparql -*-
    SELECT ?name WHERE {
        ?name a "Schedule:Compact" .
    }"""
        return list(set(r.name for r in self.query(q)))


def undirected_graph_to_dot(g: UndirectedGraph[T]) -> str:
    o = ""
    o += "graph G {\n"

    for a, neighbors in g.items():
        o += f'"{a.toPython()}";\n'

        for b in neighbors:
            if a < b:
                o += f'"{a.toPython()}" -- "{b.toPython()}";\n'
    o += "}\n"
    return o
