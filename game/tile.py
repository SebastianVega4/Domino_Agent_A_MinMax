"""
tile.py — Domino tile (ficha) representation.

A tile is an immutable pair (left, right) where both values
are integers in [0, 6] for the standard double-6 set.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Tile:
    """Immutable domino tile with two pip values."""

    left: int   # Left (or top) half
    right: int  # Right (or bottom) half

    # ------------------------------------------------------------------ #
    #  Basic properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def value(self) -> int:
        """Total pip count (used for scoring)."""
        return self.left + self.right

    @property
    def is_double(self) -> bool:
        """True if both halves are equal (e.g. [3|3])."""
        return self.left == self.right

    # ------------------------------------------------------------------ #
    #  Placement helpers                                                   #
    # ------------------------------------------------------------------ #

    def flipped(self) -> Tile:
        """Return new tile with left/right swapped."""
        return Tile(self.right, self.left)

    def can_play_on(self, end: int) -> bool:
        """Return True if this tile shares a value with *end*."""
        return self.left == end or self.right == end

    def orient_for_right_end(self, end: int) -> Optional[Tile]:
        """
        Return an oriented version of this tile so that its LEFT side
        connects to the board's current right end.

        Board right end: ...──[end|?]
        Placed tile must satisfy: tile.left == end

        Returns None if the tile cannot connect.
        """
        if self.left == end:
            return self
        if self.right == end:
            return self.flipped()
        return None

    def orient_for_left_end(self, end: int) -> Optional[Tile]:
        """
        Return an oriented version of this tile so that its RIGHT side
        connects to the board's current left end.

        Board left end: [?|end]──...
        Placed tile must satisfy: tile.right == end

        Returns None if the tile cannot connect.
        """
        if self.right == end:
            return self
        if self.left == end:
            return self.flipped()
        return None

    # ------------------------------------------------------------------ #
    #  Coordinate representation (used by heuristics)                     #
    # ------------------------------------------------------------------ #

    def as_point(self) -> tuple:
        """
        Treat the tile as a 2-D point (left, right) in pip space.
        Used by Manhattan / Euclidean distance heuristics.
        """
        return (self.left, self.right)

    def canonical(self) -> Tile:
        """Return canonical form where left <= right (for set membership)."""
        if self.left <= self.right:
            return self
        return self.flipped()

    # ------------------------------------------------------------------ #
    #  String / hash                                                       #
    # ------------------------------------------------------------------ #

    def __str__(self) -> str:
        return f"[{self.left}|{self.right}]"

    def __repr__(self) -> str:
        return self.__str__()
