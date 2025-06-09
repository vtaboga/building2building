"""This module is used to extract useful data from energyplus building files."""

import sys
import profile
import rdflib
import rdflib.namespace
import json
import urllib.parse


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def quote(n):
    if ":" in n:
        return '"' + n + '"'
    return n


ns = rdflib.Namespace("https://energyplus.net/")


def _intern_object(
    g: rdflib.Graph,
    subject,
    value,
):
    """Take a rdf graph, a rdf subject and a python dict/array and intern
    it into the ontology.

    Dictionnaries are interned using the keys as predicates, lists use
    has-elem.
    """

    if type(value) == list:
        haselem = ns.haselem
        hasindex = ns.hasindex
        for i, elem in enumerate(value):
            if type(elem) in [str, int, float]:
                name = rdflib.Literal(elem)
            elif type(elem) in [dict, list]:
                name = rdflib.BNode()
                _intern_object(g, name, elem)
            else:
                raise BaseException("erreur!")

            g.add((subject, haselem, name))
            g.add((name, hasindex, rdflib.Literal(i)))

    elif type(value) == dict:
        for k, elem in value.items():
            if type(elem) in [str, int, float]:
                name = rdflib.Literal(elem)
                g.add((subject, ns[k], name))
            elif type(elem) in [dict, list]:
                name = rdflib.BNode()
                _intern_object(g, name, elem)
            else:
                raise BaseException("erreur!")
            g.add((subject, ns[k], name))


def rdf_from_json(jsonfile):
    """Take an epJSON file path, read it and transform it into an RDF
    representation that can be queried (through .query(q: str)) in SPARQL.
    """
    isa = rdflib.namespace.RDF.type
    g = rdflib.Graph()
    g.bind("idf", ns)

    with open(jsonfile, "rb") as f:
        j = json.load(f)

    for typename, elems in j.items():
        for name, keyvals in elems.items():
            # Encode the type and name to create valid URIs
            safe_typename = urllib.parse.quote(typename)
            safe_name = urllib.parse.quote(name)
            
            rTypename = ns[safe_typename]
            rName = ns[safe_name]
            g.add((rName, isa, rTypename))
            for key, val in keyvals.items():
                if type(val) in [str, float, int]:
                    name = rdflib.Literal(val)
                    # Encode the key for the predicate
                    safe_key = urllib.parse.quote(key)
                    g.add((rName, ns[safe_key], name))
                if type(val) == list:
                    name = rdflib.BNode()
                    safe_key = urllib.parse.quote(key)
                    g.add((rName, ns[safe_key], name))
                    _intern_object(g, name, val)

    return g


def rdf_to_adjacency(g: rdflib.Graph):
    """Take a rdf representation of an idf file and return a new graph of zones
    idf:is_connected_to each other through surfaces'
    outside_boundary_condition_object properties."""

    query = """# -*- mode: sparql -*-
CONSTRUCT {
  ?src idf:is_connected_to ?dst .
} WHERE {
  ?surface1 idf:zone_name ?src .

  {
    ?surface1  (idf:outside_boundary_condition_object | ^idf:outside_boundary_condition_object)+  ?surface2 .
    ?surface2 idf:zone_name ?dst .
  } UNION {
    ?surface1 idf:outside_boundary_condition "Outdoors" .
    BIND ("Outdoors" as ?dst)
  }
  FILTER (?src < ?dst)
}
"""
    resp = g.query(query)
    return resp.graph


def rdf_to_dot(rdf: rdflib.Graph):
    """Take an rdflib graph and return its representation in the graphviz dot
    format."""

    o = ""
    o += "digraph G {\n"
    for (a, b, c) in rdf.query("""SELECT ?a ?b ?c WHERE { ?a ?b ?c . }"""):
        o += f'"{a}" -> "{c}" [label="{b}"];\n'
    o += "}\n"
    return o


def rdf_zones(rdf: rdflib.Graph) -> list[str]:
    """Return a list of all the zones in the building"""
    q = """# -*- mode: sparql -*-
SELECT ?name WHERE {
    ?name a ns:Zone .
}"""
    # Strip the URI prefix from zone names
    return list(set(str(x).replace(str(ns), '') for (x,) in rdf.query(q, initNs={"ns": ns})))


def rdf_schedules(rdf: rdflib.Graph):
    """Return a list of all the scheduler names in the building"""
    q = """# -*- mode: sparql -*-
SELECT ?name WHERE {
    ?name a ns:Schedule:Compact .
}"""
    return list(set(x.value for (x,) in rdf.query(q, initNs={"ns": ns})))
