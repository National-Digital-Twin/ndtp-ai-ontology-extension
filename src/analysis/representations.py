from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDF, RDFS, OWL


def extract_entities_from_ttl(ttl_file_path):
    """Extract entities from a TTL file and create string representations."""
    print(f"Processing {ttl_file_path}...")

    # Load the TTL file
    g = Graph()
    g.parse(ttl_file_path, format="turtle")

    entities = []

    # Extract classes
    classes = list(g.subjects(RDF.type, RDFS.Class)) + list(
        g.subjects(RDF.type, OWL.Class)
    )
    for cls in classes:
        if isinstance(cls, URIRef):
            entity_data = {
                "type": "class",
                "uri": str(cls),
                "label": get_label(g, cls),
                "comment": get_comment(g, cls),
                "subclass_of": get_subclass_of(g, cls),
                "properties": get_properties_for_class(g, cls),
            }

            # Create string representation
            entity_str = create_entity_string(entity_data)
            entities.append(entity_data | {"string_representation": entity_str})

    # Extract properties
    properties = (
        list(g.subjects(RDF.type, RDF.Property))
        + list(g.subjects(RDF.type, OWL.ObjectProperty))
        + list(g.subjects(RDF.type, OWL.DatatypeProperty))
    )

    for prop in properties:
        if isinstance(prop, URIRef):
            entity_data = {
                "type": "property",
                "uri": str(prop),
                "label": get_label(g, prop),
                "comment": get_comment(g, prop),
                "subproperty_of": get_subproperty_of(g, prop),
                "domain": get_domains(g, prop),
                "range": get_ranges(g, prop),
            }

            # Create string representation
            entity_str = create_entity_string(entity_data)
            entities.append(entity_data | {"string_representation": entity_str})

    # Extract instances (individuals)
    # This is a bit trickier as we need to filter out classes and properties
    for s, p, o in g:
        if (
            isinstance(s, URIRef)
            and p == RDF.type
            and o != RDFS.Class
            and o != OWL.Class
            and o != RDF.Property
            and o != OWL.ObjectProperty
            and o != OWL.DatatypeProperty
        ):
            # Check if this subject is not already processed as a class or property
            if not any(e["uri"] == str(s) for e in entities):
                entity_data = {
                    "type": "instance",
                    "uri": str(s),
                    "label": get_label(g, s),
                    "comment": get_comment(g, s),
                    "types": get_types(g, s),
                    "properties": get_property_values(g, s),
                }

                # Create string representation
                entity_str = create_entity_string(entity_data)
                entities.append(entity_data | {"string_representation": entity_str})

    return entities


def get_label(g, entity):
    """Get the label of an entity."""
    labels = list(g.objects(entity, RDFS.label))
    if labels:
        # Return the first label as string, removing language tag if present
        label = labels[0]
        if isinstance(label, Literal):
            return str(label.value) if hasattr(label, "value") else str(label)
        return str(label)
    return ""


def get_comment(g, entity):
    """Get the comment/description of an entity."""
    comments = list(g.objects(entity, RDFS.comment))
    if comments:
        # Return the first comment as string, removing language tag if present
        comment = comments[0]
        if isinstance(comment, Literal):
            return str(comment.value) if hasattr(comment, "value") else str(comment)
        return str(comment)
    return ""


def get_subclass_of(g, cls):
    """Get superclasses of a class."""
    return [str(o) for o in g.objects(cls, RDFS.subClassOf) if isinstance(o, URIRef)]


def get_subproperty_of(g, prop):
    """Get superproperties of a property."""
    return [
        str(o) for o in g.objects(prop, RDFS.subPropertyOf) if isinstance(o, URIRef)
    ]


def get_domains(g, prop):
    """Get domains of a property."""
    return [str(o) for o in g.objects(prop, RDFS.domain) if isinstance(o, URIRef)]


def get_ranges(g, prop):
    """Get ranges of a property."""
    return [str(o) for o in g.objects(prop, RDFS.range) if isinstance(o, URIRef)]


def get_types(g, instance):
    """Get types of an instance."""
    return [str(o) for o in g.objects(instance, RDF.type) if isinstance(o, URIRef)]


def get_properties_for_class(g, cls):
    """Get properties that have this class as domain."""
    properties = []
    for s, p, o in g.triples((None, RDFS.domain, cls)):
        if isinstance(s, URIRef):
            properties.append(str(s))
    return properties


def get_property_values(g, instance):
    """Get property-value pairs for an instance."""
    prop_values = []
    for p, o in g.predicate_objects(instance):
        if (
            isinstance(p, URIRef)
            and p != RDF.type
            and p != RDFS.label
            and p != RDFS.comment
        ):
            value = (
                str(o.value)
                if isinstance(o, Literal) and hasattr(o, "value")
                else str(o)
            )
            prop_values.append({"property": str(p), "value": value})
    return prop_values


def create_entity_string(entity_data):
    """Create a string representation of an entity for embedding."""
    entity_type = entity_data["type"]
    uri = entity_data["uri"]
    label = entity_data["label"]
    comment = entity_data["comment"]

    # Start with basic information
    parts = [f"Type: {entity_type}", f"URI: {uri}"]

    if label:
        parts.append(f"Label: {label}")

    if comment:
        parts.append(f"Description: {comment}")

    # Add type-specific information
    if entity_type == "class":
        if entity_data["subclass_of"]:
            parts.append(f"Subclass of: {', '.join(entity_data['subclass_of'])}")
        if entity_data["properties"]:
            parts.append(f"Properties: {', '.join(entity_data['properties'])}")

    elif entity_type == "property":
        if entity_data["subproperty_of"]:
            parts.append(f"Subproperty of: {', '.join(entity_data['subproperty_of'])}")
        if entity_data["domain"]:
            parts.append(f"Domain: {', '.join(entity_data['domain'])}")
        if entity_data["range"]:
            parts.append(f"Range: {', '.join(entity_data['range'])}")

    elif entity_type == "instance":
        if entity_data["types"]:
            parts.append(f"Instance of: {', '.join(entity_data['types'])}")
        if entity_data["properties"]:
            prop_value_strs = [
                f"{pv['property']} = {pv['value']}" for pv in entity_data["properties"]
            ]
            if prop_value_strs:
                parts.append(f"Property values: {'; '.join(prop_value_strs)}")

    return " | ".join(parts)
