"""Plain data containers shared by every part of the system.

Nothing in here knows about BERT, Word, or Faker. That is what lets the
models, handlers and strategies be swapped independently.
"""
from dataclasses import dataclass


@dataclass
class Entity:
    """One detected piece of sensitive text, located by character offsets."""
    start: int          # index of first character in the segment text
    end: int            # index AFTER the last character (Python slice style)
    label: str          # canonical label: "PER", "ORG", "LOC", ...
    text: str           # == segment_text[start:end]
    score: float = 1.0  # model confidence, 0..1


@dataclass
class Segment:
    """A unit of text that is sent to the NER model.

    .txt  -> one segment holding the whole file
    .docx -> one segment per paragraph / table cell / header / footer (Day 3)
    """
    id: str
    text: str


@dataclass
class Document:
    segments: list
    raw: object = None   # handler-specific data needed again at write() time
