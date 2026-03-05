"""
agent.py — High-level AI agent that combines A* and Minimax.

Strategy selection
------------------
ASTAR   : Always uses A* (good for early game planning).
MINIMAX : Always uses Minimax (good for adversarial end-game).
"""
from __future__ import annotations
from enum import Enum
from typing import Optional, Tuple

from game.tile import Tile
from game.game_state import GameState
from ai.astar import AStarSearch
from ai.minimax import MinimaxAgent
from utils.metrics import MetricsTracker, MoveRecord

# Switch to Minimax when player holds ≤ this many tiles
SWITCH_THRESHOLD = 4


class Strategy(Enum):
    ASTAR   = "A*"
    MINIMAX = "Minimax + alpha-beta"


class DominoAgent:
    """
    AI agent for 2-player domino.

    Parameters
    ----------
    player_id     : 0 or 1.
    strategy      : Algorithm selection policy.
    minimax_depth : Ply depth for Minimax search.
    euclidean     : Use Euclidean instead of Manhattan heuristic.
    alpha_beta    : Enable Alpha-Beta pruning in Minimax.
    """

    def __init__(
        self,
        player_id: int,
        strategy: Strategy = Strategy.MINIMAX,
        minimax_depth: int = 4,
        euclidean: bool = False,
        alpha_beta: bool = True,
        name: str = "",
    ) -> None:
        self.player_id = player_id
        self.strategy = strategy
        self.name = name or f"Agente-{player_id}"

        self._astar = AStarSearch(
            player_id=player_id,
            euclidean=euclidean,
            max_depth=8,
            time_limit=1.5,
        )
        self._minimax = MinimaxAgent(
            player_id=player_id,
            max_depth=minimax_depth,
            time_limit=1.5,
            alpha_beta=alpha_beta,
            euclidean=euclidean,
        )

        self.metrics = MetricsTracker()
        self.move_count = 0
        self.last_tree = None
        self.last_algo_name = ""

    # ------------------------------------------------------------------ #
    #  Core decision                                                       #
    # ------------------------------------------------------------------ #

    def decide(self, state: GameState) -> Tuple[Optional[Tuple[Tile, str]], float]:
        """
        Choose the best move from *state*.
        Returns ((tile, side), score) or (None, 0.0) if no valid moves exist.
        """
        moves = state.get_valid_moves()
        if not moves:
            return None, 0.0
        if len(moves) == 1:
            self._record(state, algorithm_used=self.strategy.value)
            return moves[0], 0.0

        algorithm_used = "A*" if self.strategy == Strategy.ASTAR else "Minimax"
        move = self._execute(state, algorithm_used)
        
        # Get score from the specific algorithm
        score = 0.0
        if algorithm_used == "Minimax" and self._minimax.last_tree:
            score = self._minimax.last_tree.score

        self.move_count += 1
        self._record(state, algorithm_used)

        return (move or moves[0]), score



    def _execute(
        self, state: GameState, algorithm: str
    ) -> Optional[Tuple[Tile, str]]:
        if algorithm == "A*":
            move, tree = self._astar.find_best_move(state)
            self.last_tree = tree
            self.last_algo_name = "A*"
            return move
        
        move = self._minimax.find_best_move(state)
        self.last_tree = self._minimax.last_tree
        self.last_algo_name = "Minimax"
        return move

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def _record(self, state: GameState, algorithm_used: str) -> None:
        rec = MoveRecord(
            algorithm=algorithm_used,
            astar_nodes_expanded=self._astar.nodes_expanded,
            astar_nodes_generated=self._astar.nodes_generated,
            astar_time=self._astar.execution_time,
            minimax_nodes=self._minimax.nodes_expanded,
            minimax_pruned=self._minimax.nodes_pruned,
            minimax_time=self._minimax.execution_time,
            hand_size=len(state.hands[self.player_id]),
            hand_value=state.hand_value(self.player_id),
        )
        self.metrics.record_move(rec)

    def record_game_result(self, won: bool, final_hand_value: int, turns: int) -> None:
        self.metrics.record_game(won=won, final_hand_value=final_hand_value, turns=turns)

    # ------------------------------------------------------------------ #
    #  Diagnostic helpers                                                  #
    # ------------------------------------------------------------------ #

    @property
    def last_astar_nodes(self) -> int:
        return self._astar.nodes_expanded

    @property
    def last_minimax_nodes(self) -> int:
        return self._minimax.nodes_expanded

    @property
    def last_astar_time(self) -> float:
        return self._astar.execution_time

    @property
    def last_minimax_time(self) -> float:
        return self._minimax.execution_time

    @property
    def last_evaluations(self) -> list:
        return self._minimax.last_evaluations

    def summary(self) -> dict:
        return self.metrics.summary()

    def __repr__(self) -> str:
        return f"DominoAgent(id={self.player_id}, strategy={self.strategy.value})"
