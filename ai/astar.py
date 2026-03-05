"""
astar.py — A* search for single-agent domino planning.

Problem formulation
-------------------
The agent treats its own turn sequence as a shortest-path problem:

  State   : (board.left_end, board.right_end, frozenset(hand), depth)
  Start   : current game state
  Goal    : any state where the player's hand is empty
  g(n)    : number of tiles the agent has played so far
  h(n)    : Heuristics.hand_distance() — Manhattan or Euclidean
            aggregate distance of remaining tiles to board ends

The opponent is handled pessimistically: when it is the opponent's
turn the search assumes a 'neutral pass' (board ends unchanged) so
that we plan only for our own moves.  This is a deliberate
simplification: full two-player A* would be Bidirectional Minimax,
which overlaps with the Minimax module.

The returned best_first_move is the first move of the optimal
(minimum-f) path found within the time/depth budget.
"""
from __future__ import annotations
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from game.tile import Tile
from game.game_state import GameState
from ai.heuristics import Heuristics


# ─── Node ──────────────────────────────────────────────────────────────── #

@dataclass(order=True)
class _Node:
    """Priority-queue entry for A*."""
    f: float                                          # f = g + h (sort key)
    g: float          = field(compare=False)          # cost so far
    state: GameState  = field(compare=False)
    path: list        = field(compare=False)          # list of (Tile, side)
    nodes: list       = field(compare=False, default_factory=list) # List of MinimaxNodes for visualization


# ─── AStarSearch ────────────────────────────────────────────────────────── #

class AStarSearch:
    """
    A* planner for a single domino player.

    Parameters
    ----------
    player_id    : The agent's player index (0 or 1).
    euclidean    : Use Euclidean distance heuristic (default: Manhattan).
    max_depth    : Maximum number of our own moves to look ahead.
    time_limit   : Wall-clock budget in seconds.
    """

    def __init__(
        self,
        player_id: int,
        euclidean: bool = False,
        max_depth: int = 8,
        time_limit: float = 1.5,
    ) -> None:
        self.player_id = player_id
        self.euclidean = euclidean
        self.max_depth = max_depth
        self.time_limit = time_limit

        # Diagnostics (reset each call to find_best_move)
        self.nodes_expanded: int = 0
        self.nodes_generated: int = 0
        self.execution_time: float = 0.0
        self.path_found: bool = False

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def find_best_move(
        self, state: GameState, record_tree: bool = True
    ) -> Tuple[Optional[Tuple[Tile, str]], Optional[MinimaxNode]]:
        """
        Run A* from *state* and return the best first move
        (tile, side) for the current player, or None if no moves exist.
        """
        t0 = time.perf_counter()
        self.nodes_expanded = 0
        self.nodes_generated = 0
        self.path_found = False

        moves = state.get_valid_moves()
        if not moves:
            self.execution_time = time.perf_counter() - t0
            return None
        if len(moves) == 1:
            self.execution_time = time.perf_counter() - t0
            return moves[0]

        result, tree = self._search(state, t0, record_tree)
        self.execution_time = time.perf_counter() - t0
        return result, tree

    # ------------------------------------------------------------------ #
    #  Core search                                                         #
    # ------------------------------------------------------------------ #

    def _h(self, state: GameState) -> float:
        """h(n): admissibility-motivated heuristic."""
        hand = state.hands[self.player_id]
        if not hand:
            return 0.0
        if state.board.is_empty():
            # No board yet — use raw pip sum / max_pip as proxy
            return sum(t.value for t in hand) / (2 * 6)
        return Heuristics.hand_distance(
            hand,
            state.board.left_end,
            state.board.right_end,
            self.euclidean,
        )

    def _search(self, initial: GameState, t0: float, record: bool) -> Tuple[Optional[Tuple[Tile, str]], Optional[MinimaxNode]]:
        from ai.minimax import MinimaxNode
        open_heap: List[_Node] = []
        visited: Dict[tuple, float] = {}

        h0 = self._h(initial)
        root_node_vis = MinimaxNode(
            label="RAIZ (A*)", score=h0, depth=0, is_max=True,
            eval_details={"h(n)": h0}
        ) if record else None

        start_node = _Node(f=h0, g=0.0, state=initial, path=[], nodes=[root_node_vis] if root_node_vis else [])
        heapq.heappush(open_heap, start_node)
        self.nodes_generated += 1

        best_node_obj: Optional[_Node] = start_node
        best_f = h0

        while open_heap:
            if time.perf_counter() - t0 > self.time_limit:
                break

            node = heapq.heappop(open_heap)
            self.nodes_expanded += 1

            sh = node.state.to_hash()
            if sh in visited and visited[sh] <= node.g:
                continue
            visited[sh] = node.g

            # ── Goal check ──────────────────────────────────────── #
            if len(node.state.hands[self.player_id]) == 0:
                if node.path:
                    self.path_found = True
                    return node.path[0]   # Optimal first move
                break

            # ── Depth / terminal limit ───────────────────────────── #
            our_moves_made = len(node.path)
            if node.state.is_terminal() or our_moves_made >= self.max_depth:
                if node.f < best_f and node.path:
                    best_f = node.f
                    best_first = node.path[0]
                continue

            # ── Expand ──────────────────────────────────────────── #
            if node.state.current_player == self.player_id:
                # Our turn — try every valid move
                for tile, side in node.state.get_valid_moves():
                    nxt = node.state.apply_move(tile, side)
                    g_new = node.g + 1.0
                    h_new = self._h(nxt)
                    f_new = g_new + h_new
                    
                    child_vis = None
                    if record:
                        lbl = f"{tile}→{'L' if side=='left' else 'R'}"
                        child_vis = MinimaxNode(
                            move=(tile, side), score=f_new, 
                            depth=node.nodes[-1].depth + 1 if node.nodes else 1,
                            is_max=True, label=lbl, eval_details={"g(n)": g_new, "h(n)": h_new}
                        )
                        if node.nodes: node.nodes[-1].children.append(child_vis)

                    new_path = node.path + [(tile, side)]
                    child = _Node(f=f_new, g=g_new, state=nxt, path=new_path, nodes=node.nodes + ([child_vis] if child_vis else []))
                    heapq.heappush(open_heap, child)
                    self.nodes_generated += 1

                    if f_new < best_f and new_path:
                        best_f = f_new
                        best_node_obj = child

            else:
                # Opponent's turn — apply neutral pass (pessimistic)
                nxt = node.state.apply_pass()
                g_new = node.g          # Does not cost us anything
                h_new = self._h(nxt)
                f_new = g_new + h_new
                
                child_vis = None
                if record:
                    child_vis = MinimaxNode(
                        label="Oponente (Pass)", score=f_new, 
                        depth=node.nodes[-1].depth + 1 if node.nodes else 1,
                        is_max=False, eval_details={"g(n)": g_new, "h(n)": h_new}
                    )
                    if node.nodes: node.nodes[-1].children.append(child_vis)

                child = _Node(f=f_new, g=g_new, state=nxt, path=node.path, nodes=node.nodes + ([child_vis] if child_vis else []))
                heapq.heappush(open_heap, child)
                self.nodes_generated += 1

        # Al terminar, si encontramos un camino, el primer movimiento es path[0]
        # El arbol para visualizacion sera la secuencia de la ruta elegida (A* visualizado como un camino lineal)
        final_tree = best_node_obj.nodes[0] if best_node_obj and best_node_obj.nodes else None
        # Marcar la ruta elegida
        if best_node_obj and best_node_obj.nodes:
            for n in best_node_obj.nodes:
                n.is_chosen = True
        
        move = best_node_obj.path[0] if best_node_obj and best_node_obj.path else self._greedy_fallback(initial)
        return move, final_tree

    # ------------------------------------------------------------------ #
    #  Greedy fallback                                                     #
    # ------------------------------------------------------------------ #

    def _greedy_fallback(self, state: GameState) -> Optional[Tuple[Tile, str]]:
        """
        One-step greedy: pick the move that minimises h(next_state).
        Used when A* exhausts its budget without finding a complete path.
        """
        moves = state.get_valid_moves()
        if not moves:
            return None

        # Edge case: empty board → prefer highest double, else highest value
        if state.board.is_empty():
            doubles = [(t, s) for t, s in moves if t.is_double]
            pool = doubles if doubles else moves
            return max(pool, key=lambda m: m[0].value)

        def score(move: Tuple[Tile, str]) -> float:
            tile, side = move
            nxt = state.apply_move(tile, side)
            return self._h(nxt)

        return min(moves, key=score)
