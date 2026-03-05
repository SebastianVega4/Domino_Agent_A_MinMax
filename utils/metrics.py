"""
metrics.py — Performance metrics collection and reporting.

Collects per-move and per-game statistics from the AI agents
so that results can be tabulated and plotted for the IEEE report.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class MoveRecord:
    """Single-move performance snapshot."""
    algorithm: str
    astar_nodes_expanded: int
    astar_nodes_generated: int
    astar_time: float          # seconds
    minimax_nodes: int
    minimax_pruned: int
    minimax_time: float        # seconds
    hand_size: int
    hand_value: int


@dataclass
class GameRecord:
    """End-of-game summary."""
    won: bool
    final_hand_value: int
    turns: int


class MetricsTracker:
    """Accumulates move- and game-level metrics for one agent."""

    def __init__(self) -> None:
        self.moves:  List[MoveRecord] = []
        self.games:  List[GameRecord] = []

    # ------------------------------------------------------------------ #
    #  Recording                                                           #
    # ------------------------------------------------------------------ #

    def record_move(self, rec: MoveRecord) -> None:
        self.moves.append(rec)

    def record_game(self, won: bool, final_hand_value: int, turns: int) -> None:
        self.games.append(GameRecord(won=won, final_hand_value=final_hand_value, turns=turns))

    def reset(self) -> None:
        self.moves.clear()
        self.games.clear()

    # ------------------------------------------------------------------ #
    #  Aggregation helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_mean(values: List[float]) -> float:
        return statistics.mean(values) if values else 0.0

    @staticmethod
    def _safe_stdev(values: List[float]) -> float:
        return statistics.stdev(values) if len(values) > 1 else 0.0

    # ------------------------------------------------------------------ #
    #  Summary                                                             #
    # ------------------------------------------------------------------ #

    def summary(self) -> Dict[str, Any]:
        """Return a flat dict with all key statistics."""
        astar_times   = [m.astar_time   for m in self.moves if m.astar_time   > 0]
        minimax_times = [m.minimax_time for m in self.moves if m.minimax_time > 0]
        astar_exp     = [m.astar_nodes_expanded  for m in self.moves if m.astar_nodes_expanded  > 0]
        astar_gen     = [m.astar_nodes_generated for m in self.moves if m.astar_nodes_generated > 0]
        mm_nodes      = [m.minimax_nodes  for m in self.moves if m.minimax_nodes  > 0]
        mm_pruned     = [m.minimax_pruned for m in self.moves if m.minimax_pruned > 0]

        wins = sum(1 for g in self.games if g.won)
        avg_turns = self._safe_mean([g.turns for g in self.games])

        return {
            # A* stats
            "astar_calls":              len(astar_times),
            "astar_avg_time_ms":        self._safe_mean(astar_times) * 1000,
            "astar_std_time_ms":        self._safe_stdev(astar_times) * 1000,
            "astar_avg_nodes_expanded": self._safe_mean(astar_exp),
            "astar_avg_nodes_generated":self._safe_mean(astar_gen),
            # Minimax stats
            "minimax_calls":            len(minimax_times),
            "minimax_avg_time_ms":      self._safe_mean(minimax_times) * 1000,
            "minimax_std_time_ms":      self._safe_stdev(minimax_times) * 1000,
            "minimax_avg_nodes":        self._safe_mean(mm_nodes),
            "minimax_avg_pruned":       self._safe_mean(mm_pruned),
            # Game stats
            "games_played":             len(self.games),
            "wins":                     wins,
            "win_rate":                 wins / len(self.games) if self.games else 0.0,
            "avg_turns_per_game":       avg_turns,
            "total_moves_recorded":     len(self.moves),
        }

    def per_algorithm_breakdown(self) -> Dict[str, Dict[str, float]]:
        """
        Break down average node count and time per algorithm label.
        Useful for comparison tables in the report.
        """
        groups: Dict[str, List[MoveRecord]] = {}
        for m in self.moves:
            groups.setdefault(m.algorithm, []).append(m)

        result = {}
        for alg, recs in groups.items():
            t_list = [r.astar_time + r.minimax_time for r in recs]
            n_list = [r.astar_nodes_expanded + r.minimax_nodes for r in recs]
            result[alg] = {
                "count": len(recs),
                "avg_time_ms": self._safe_mean(t_list) * 1000,
                "avg_nodes":   self._safe_mean(n_list),
            }
        return result

    def print_report(self, agent_name: str = "Agente") -> None:
        """Print a formatted metrics report to stdout."""
        s = self.summary()
        sep = "-" * 52
        print(f"\n{sep}")
        print(f"  Reporte de Metricas - {agent_name}")
        print(sep)
        print(f"  Partidas jugadas   : {s['games_played']}")
        print(f"  Victorias          : {s['wins']}  ({s['win_rate']*100:.1f}%)")
        print(f"  Turnos promedio    : {s['avg_turns_per_game']:.1f}")
        print(sep)
        print(f"  A*  - llamadas     : {s['astar_calls']}")
        print(f"       tiempo prom.  : {s['astar_avg_time_ms']:.2f} ms  (sigma={s['astar_std_time_ms']:.2f})")
        print(f"       nodos expan.  : {s['astar_avg_nodes_expanded']:.1f}")
        print(f"       nodos gen.    : {s['astar_avg_nodes_generated']:.1f}")
        print(sep)
        print(f"  MM  - llamadas     : {s['minimax_calls']}")
        print(f"       tiempo prom.  : {s['minimax_avg_time_ms']:.2f} ms  (sigma={s['minimax_std_time_ms']:.2f})")
        print(f"       nodos expan.  : {s['minimax_avg_nodes']:.1f}")
        print(f"       nodos podados : {s['minimax_avg_pruned']:.1f}")
        print(sep)
