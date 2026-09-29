from .model import Schema, EntityType, RelationType, Attribute, Axiom, SchemaParseError
from .owlcheck import check_schema

__all__ = [
    "Schema", "EntityType", "RelationType", "Attribute", "Axiom", "SchemaParseError",
    "check_schema",
]
