"""
game_engine.py — Game initialization, rule enforcement, and utilities.
"""
from __future__ import annotations
import random
from typing import List, Tuple, Optional

from game.tile import Tile
from game.board import Board
from game.game_state import GameState

# Standard double-6 set parameters
MAX_PIP = 6
TILES_PER_PLAYER = 7
NUM_PLAYERS = 2


class DominoEngine:
    """Static factory / utility class for domino game management."""

    # ------------------------------------------------------------------ #
    #  Tile-set generation                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def create_tileset() -> List[Tile]:
        """
        Generate the complete double-6 set: 28 tiles.
        (i, j) for 0 ≤ i ≤ j ≤ 6.
        """
        return [Tile(i, j) for i in range(MAX_PIP + 1) for j in range(i, MAX_PIP + 1)]

    @staticmethod
    def deal(
        tileset: List[Tile],
        num_players: int = NUM_PLAYERS,
        per_player: int = TILES_PER_PLAYER,
        seed: Optional[int] = None,
    ) -> Tuple[List[List[Tile]], List[Tile]]:
        """
        Shuffle *tileset* and deal *per_player* tiles to each player.
        Returns (hands, boneyard).
        """
        deck = list(tileset)
        rng = random.Random(seed)
        rng.shuffle(deck)

        hands: List[List[Tile]] = []
        for _ in range(num_players):
            hands.append(deck[:per_player])
            deck = deck[per_player:]

        return hands, deck   # deck remainder = boneyard

    # ------------------------------------------------------------------ #
    #  First-player determination                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def first_player(hands: List[List[Tile]]) -> Tuple[int, Optional[Tile]]:
        """
        Classic rule: the player holding the highest double opens the game.
        If no doubles, the player with the highest-value tile goes first.

        Returns (player_index, opening_tile_or_None).
        """
        best_double_val = -1
        best_player = 0
        best_tile: Optional[Tile] = None

        for idx, hand in enumerate(hands):
            for tile in hand:
                if tile.is_double and tile.left > best_double_val:
                    best_double_val = tile.left
                    best_player = idx
                    best_tile = tile

        if best_double_val == -1:
            # No doubles — highest pip sum
            best_val = -1
            for idx, hand in enumerate(hands):
                for tile in hand:
                    if tile.value > best_val:
                        best_val = tile.value
                        best_player = idx
                        best_tile = tile

        return best_player, best_tile

    # ------------------------------------------------------------------ #
    #  Full game-state factory                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def new_game(seed: Optional[int] = None) -> GameState:
        """
        Create a freshly shuffled, fully dealt initial GameState.
        The first player is set according to the highest-double rule.
        """
        tileset = DominoEngine.create_tileset()
        hands, boneyard = DominoEngine.deal(tileset, seed=seed)
        first, _ = DominoEngine.first_player(hands)

        return GameState(
            board=Board(),
            hands=hands,
            current_player=first,
            boneyard=boneyard,
            pass_count=0,
        )

    # ------------------------------------------------------------------ #
    #  Rule helpers                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def must_draw(state: GameState) -> bool:
        """
        True if the current player cannot play but may draw from the boneyard.
        """
        has_moves = bool(state.get_valid_moves())
        return (not has_moves) and bool(state.boneyard)

    @staticmethod
    def must_pass(state: GameState) -> bool:
        """
        True if the current player cannot play and the boneyard is empty.
        """
        has_moves = bool(state.get_valid_moves())
        return (not has_moves) and (not state.boneyard)

    @staticmethod
    def score_summary(state: GameState) -> dict:
        """
        Return a summary dict with hand values and winner info.
        """
        return {
            "hand_value_0": state.hand_value(0),
            "hand_value_1": state.hand_value(1),
            "winner": state.winner(),
            "is_terminal": state.is_terminal(),
            "board_tiles": len(state.board),
        }
