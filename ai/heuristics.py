"""
heuristics.py — Heuristic functions for the domino AI agent.

Conceptual mapping to distance metrics
---------------------------------------
Each domino tile (a, b) is treated as a point P = (a, b) in ℝ².
A board end value *e* maps to the diagonal point Q = (e, e).

Manhattan distance  :  d₁(P, Q) = |a − e| + |b − e|
Euclidean distance  :  d₂(P, Q) = √[(a − e)² + (b − e)²]

Interpretation
--------------
- Distance = 0  ⟹  tile is immediately playable on that end.
- Higher distance ⟹  tile is "farther" from the current board ends,
  so more intermediate moves are required before it can be placed.

These distances feed into A* as the admissibility-motivated h(n)
estimate and into Minimax as part of the evaluation function.
"""
from __future__ import annotations
import math
from typing import List

from game.tile import Tile


class Heuristics:
    """Static collection of heuristic functions."""

    # ------------------------------------------------------------------ #
    #  Core distance functions                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def manhattan(tile: Tile, end: int) -> float:
        """
        Manhattan distance between tile-point (a, b) and diagonal-point (e, e).

        d₁ = |a − e| + |b − e|

        We take the minimum over both orientations because the tile can
        be flipped: min(d(left→e, right→e), d(right→e, left→e)).
        Since addition is commutative the result is always the same, but
        the explicit minimum makes the intent clear.
        """
        return float(abs(tile.left - end) + abs(tile.right - end))

    @staticmethod
    def euclidean(tile: Tile, end: int) -> float:
        """
        Euclidean distance between tile-point (a, b) and diagonal-point (e, e).

        d₂ = √[(a − e)² + (b − e)²]
        """
        return math.sqrt((tile.left - end) ** 2 + (tile.right - end) ** 2)

    # ------------------------------------------------------------------ #
    #  Tile-level compatibility score                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def tile_distance(
        tile: Tile,
        left_end: int,
        right_end: int,
        euclidean: bool = False,
    ) -> float:
        """
        Minimum distance from *tile* to either board end.

        Returns 0.0 if the tile is immediately playable.
        """
        if tile.can_play_on(left_end) or tile.can_play_on(right_end):
            return 0.0

        dist_fn = Heuristics.euclidean if euclidean else Heuristics.manhattan

        return min(
            dist_fn(tile, left_end),
            dist_fn(tile, right_end),
        )

    # ------------------------------------------------------------------ #
    #  Hand-level h(n) estimate                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def hand_distance(
        hand: List[Tile],
        left_end: int,
        right_end: int,
        euclidean: bool = False,
    ) -> float:
        """
        Aggregate heuristic for a player's hand.

        h(n) = Σ tile_distance(t, left_end, right_end) for t in hand.

        Semantics for A*
        ----------------
        - g(n) counts tiles already played (moves made).
        - h(n) estimates the "cost" to become playable for each
          remaining tile.  A tile already playable costs 0; an
          incompatible tile costs its minimum distance, which acts as a
          lower bound on the number of board-end-change events needed
          before the tile can be played.
        """
        if not hand:
            return 0.0
        return sum(
            Heuristics.tile_distance(t, left_end, right_end, euclidean)
            for t in hand
        )

    # ------------------------------------------------------------------ #
    #  Auxiliary evaluation signals (used by Minimax eval function)       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def playability_ratio(hand: List[Tile], left_end: int, right_end: int) -> float:
        """
        Fraction of hand tiles that are immediately playable.
        Range [0, 1].  Higher = better board control.
        """
        if not hand:
            return 1.0
        playable = sum(
            1 for t in hand if t.can_play_on(left_end) or t.can_play_on(right_end)
        )
        return playable / len(hand)

    @staticmethod
    def hand_pip_value(hand: List[Tile]) -> int:
        """Total pip count of a hand (lower = better for winning)."""
        return sum(t.value for t in hand)

    @staticmethod
    def blocking_potential(
        tile: Tile,
        hand: List[Tile],
        left_end: int,
        right_end: int,
    ) -> float:
        """
        After playing *tile*, estimate what fraction of remaining hand
        is still playable.  Used to prefer moves that keep options open.
        """
        remaining = [t for t in hand if t.canonical() != tile.canonical()]
        if not remaining:
            return 1.0
        return Heuristics.playability_ratio(remaining, left_end, right_end)
