"""
game_state.py — Immutable-style game state snapshot.

Designed for use in A* and Minimax search trees: cloning is cheap
and all transition methods return new states.
"""
from __future__ import annotations
from typing import List, Tuple, Optional

from game.tile import Tile
from game.board import Board, Move


class GameState:
    """
    Complete snapshot of one point in a 2-player domino game.

    Attributes
    ----------
    board          : Current board chain.
    hands          : hands[i] is the list of tiles held by player i.
    current_player : Whose turn it is (0 or 1).
    boneyard       : Remaining undealt tiles available for drawing.
    pass_count     : Consecutive pass count (2 => blocked game).
    """

    def __init__(
        self,
        board: Board,
        hands: List[List[Tile]],
        current_player: int,
        boneyard: List[Tile],
        pass_count: int = 0,
    ) -> None:
        self.board = board
        self.hands = hands
        self.current_player = current_player
        self.boneyard = boneyard
        self.pass_count = pass_count

    # ------------------------------------------------------------------ #
    #  Move generation                                                     #
    # ------------------------------------------------------------------ #

    def get_valid_moves(self) -> List[Move]:
        """Valid moves for the current player."""
        return self.board.get_valid_moves(self.hands[self.current_player])

    # ------------------------------------------------------------------ #
    #  Transitions                                                         #
    # ------------------------------------------------------------------ #

    def apply_move(self, tile: Tile, side: str) -> GameState:
        """
        Return a new state after the current player plays *tile* on *side*.
        The tile is removed from the player's hand and placed on the board.
        """
        ns = self.clone()
        hand = ns.hands[ns.current_player]

        # Remove the tile (match by canonical form to handle orientation)
        target = tile.canonical()
        for i, t in enumerate(hand):
            if t.canonical() == target:
                hand.pop(i)
                break

        ns.board.place(tile, side)
        ns.pass_count = 0
        ns.current_player = 1 - ns.current_player
        return ns

    def apply_draw(self) -> GameState:
        """
        Draw one tile from the boneyard for the current player.
        Does NOT advance the turn — the player may then play or pass.
        """
        ns = self.clone()
        if ns.boneyard:
            drawn = ns.boneyard.pop(0)
            ns.hands[ns.current_player].append(drawn)
        return ns

    def apply_pass(self) -> GameState:
        """
        Pass the turn.  Increments pass_count and advances to the
        next player.  Call *after* draw attempts are exhausted.
        """
        ns = self.clone()
        ns.pass_count += 1
        ns.current_player = 1 - ns.current_player
        return ns

    # ------------------------------------------------------------------ #
    #  Terminal checks                                                     #
    # ------------------------------------------------------------------ #

    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        if any(len(h) == 0 for h in self.hands):
            return True           # A player emptied their hand
        if self.pass_count >= 2:
            return True           # Both players blocked
        return False

    def winner(self) -> Optional[int]:
        """
        Return the index of the winning player, or None if the game
        is not over or ends in a draw.
        """
        if len(self.hands[0]) == 0:
            return 0
        if len(self.hands[1]) == 0:
            return 1
        if self.pass_count >= 2:
            v0 = self.hand_value(0)
            v1 = self.hand_value(1)
            if v0 < v1:
                return 0
            if v1 < v0:
                return 1
            return None   # Perfect draw
        return None

    # ------------------------------------------------------------------ #
    #  Scoring helpers                                                     #
    # ------------------------------------------------------------------ #

    def hand_value(self, player: int) -> int:
        """Total pip value of *player*'s hand."""
        return sum(t.value for t in self.hands[player])

    def hand_size(self, player: int) -> int:
        return len(self.hands[player])

    # ------------------------------------------------------------------ #
    #  Hashing (for A* visited set)                                       #
    # ------------------------------------------------------------------ #

    def to_hash(self) -> tuple:
        """
        Compact, hashable fingerprint.
        Encodes board ends, board length, both hands, and turn.
        """
        return (
            self.board.left_end,
            self.board.right_end,
            len(self.board),
            frozenset(t.canonical().as_point() for t in self.hands[0]),
            frozenset(t.canonical().as_point() for t in self.hands[1]),
            self.current_player,
        )

    # ------------------------------------------------------------------ #
    #  Cloning                                                             #
    # ------------------------------------------------------------------ #

    def clone(self) -> GameState:
        return GameState(
            board=self.board.clone(),
            hands=[list(h) for h in self.hands],
            current_player=self.current_player,
            boneyard=list(self.boneyard),
            pass_count=self.pass_count,
        )

    def __repr__(self) -> str:
        return (
            f"GameState(turn={self.current_player}, "
            f"hand0={self.hands[0]}, hand1={self.hands[1]}, "
            f"passes={self.pass_count})"
        )
