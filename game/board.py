"""
board.py — The domino board (chain of placed tiles).

The board is a doubly-open chain.  Tiles can be added to the
LEFT end or the RIGHT end, provided the connecting pip values match.
"""
from __future__ import annotations
from typing import List, Tuple, Optional
import copy

from game.tile import Tile

# Move type alias
Move = Tuple[Tile, str]   # (tile, 'left' | 'right')


class Board:
    """Manages the linear chain of placed domino tiles."""

    def __init__(self) -> None:
        self.tiles: List[Tile] = []   # Tiles as placed (oriented)
        self.left_end: Optional[int] = None
        self.right_end: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Queries                                                             #
    # ------------------------------------------------------------------ #

    def is_empty(self) -> bool:
        return len(self.tiles) == 0

    def can_place_left(self, tile: Tile) -> bool:
        if self.is_empty():
            return True
        return tile.can_play_on(self.left_end)

    def can_place_right(self, tile: Tile) -> bool:
        if self.is_empty():
            return True
        return tile.can_play_on(self.right_end)

    def get_valid_moves(self, hand: List[Tile]) -> List[Move]:
        """
        Return all legal (tile, side) pairs for *hand* against
        the current board.  Duplicate (tile, left) + (tile, right)
        entries are allowed when a tile fits both ends and both ends
        differ — they represent genuinely different board states.
        """
        if self.is_empty():
            # Any tile may start the chain; expose only 'right' to avoid
            # duplicates (direction is arbitrary for the first tile).
            return [(t, 'right') for t in hand]

        moves: List[Move] = []
        seen: set = set()

        for tile in hand:
            key = tile.canonical()

            can_l = self.can_place_left(tile)
            can_r = self.can_place_right(tile)

            # If both ends carry the same value, placing left or right
            # produces identical board states — emit only one.
            if can_l and can_r and self.left_end == self.right_end:
                mk = (key, 'right')
                if mk not in seen:
                    seen.add(mk)
                    moves.append((tile, 'right'))
            else:
                if can_l:
                    mk = (key, 'left')
                    if mk not in seen:
                        seen.add(mk)
                        moves.append((tile, 'left'))
                if can_r:
                    mk = (key, 'right')
                    if mk not in seen:
                        seen.add(mk)
                        moves.append((tile, 'right'))

        return moves

    # ------------------------------------------------------------------ #
    #  Mutation                                                            #
    # ------------------------------------------------------------------ #

    def place(self, tile: Tile, side: str) -> bool:
        """
        Attempt to place *tile* on *side* ('left' or 'right').
        Returns True on success, False if the placement is illegal.
        """
        if self.is_empty():
            self.tiles.append(tile)
            self.left_end = tile.left
            self.right_end = tile.right
            return True

        if side == 'left':
            oriented = tile.orient_for_left_end(self.left_end)
            if oriented is None:
                return False
            self.tiles.insert(0, oriented)
            self.left_end = oriented.left
            return True

        elif side == 'right':
            oriented = tile.orient_for_right_end(self.right_end)
            if oriented is None:
                return False
            self.tiles.append(oriented)
            self.right_end = oriented.right
            return True

        return False

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    @property
    def pip_total(self) -> int:
        """Sum of all pips on the board."""
        return sum(t.value for t in self.tiles)

    def clone(self) -> Board:
        b = Board()
        b.tiles = list(self.tiles)
        b.left_end = self.left_end
        b.right_end = self.right_end
        return b

    def __len__(self) -> int:
        return len(self.tiles)

    def __repr__(self) -> str:
        if self.is_empty():
            return "<empty board>"
        chain = " ".join(str(t) for t in self.tiles)
        return f"L:{self.left_end}  {chain}  R:{self.right_end}"
