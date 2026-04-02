"""
minimax.py — Minimax con poda Alpha-Beta + grabación del árbol de decisión.

El árbol se almacena en self.last_tree tras cada llamada a find_best_move().
La ruta óptima se marca con is_chosen=True en cada nodo del camino ganador.
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from game.tile import Tile
from game.game_state import GameState
from ai.heuristics import Heuristics

_W_VALUE   = 0.40
_W_PLAY    = 0.30
_W_DIST    = 0.20
_W_SIZE    = 0.10
_WIN_SCORE  =  1_000.0
_LOSS_SCORE = -1_000.0


# ═══════════════════════════════════════════════════════════════════════════ #
#  Nodo del árbol Minimax                                                     #
# ═══════════════════════════════════════════════════════════════════════════ #

@dataclass
class MinimaxNode:
    """
    Nodo del árbol de búsqueda Minimax para visualización.

    Atributos semánticos
    --------------------
    move        : (Tile, side) jugado para llegar aquí, o None si es raíz.
    score       : Puntuación asignada por minimax a este nodo.
    depth       : Profundidad en el árbol (raíz = 0).
    is_max      : True → capa MAX (agente); False → capa MIN (oponente).
    is_terminal : Nodo hoja (profundidad 0 o estado terminal).
    is_chosen   : Pertenece a la ruta óptima elegida.
    was_pruned  : Subárbol podado por α-β (no completamente explorado).
    alpha_in    : Valor de α al entrar en este nodo.
    beta_in     : Valor de β al entrar en este nodo.
    label       : Etiqueta corta para la UI.
    children    : Hijos explorados.

    Atributos de layout (asignados por TreeViewer)
    -----------------------------------------------
    _x, _y       : Posición en píxeles del nodo en el lienzo virtual.
    _subtree_w   : Anchura del subárbol (usada para el posicionado).
    """
    move:        Optional[Tuple[Tile, str]] = None
    score:       float                     = 0.0
    depth:       int                       = 0
    is_max:      bool                      = True
    is_terminal: bool                      = False
    is_chosen:   bool                      = False
    was_pruned:  bool                      = False
    alpha_in:    float                     = field(default_factory=lambda: -math.inf)
    beta_in:     float                     = field(default_factory=lambda:  math.inf)
    label:       str                       = "RAÍZ"
    children:    List[MinimaxNode]          = field(default_factory=list)
    eval_details: Dict[str, float]          = field(default_factory=dict)

    # Layout (escritos por TreeViewer.compute_layout)
    _x:         float = field(default=0.0, repr=False)
    _y:         float = field(default=0.0, repr=False)
    _subtree_w: float = field(default=0.0, repr=False)

    # ── Propiedades de visualización ───────────────────────────────────── #

    @property
    def score_str(self) -> str:
        if self.score >= 999:   return "WIN OK"
        if self.score <= -999:  return "LOSS X"
        return f"{self.score:+.1f}"

    @property
    def type_label(self) -> str:
        return "MAX [^]" if self.is_max else "MIN [v]"

    @property
    def alpha_str(self) -> str:
        return "-∞" if math.isinf(self.alpha_in) and self.alpha_in < 0 else f"{self.alpha_in:+.1f}"

    @property
    def beta_str(self) -> str:
        return "+∞" if math.isinf(self.beta_in) and self.beta_in > 0 else f"{self.beta_in:+.1f}"


# ═══════════════════════════════════════════════════════════════════════════ #
#  Agente Minimax                                                             #
# ═══════════════════════════════════════════════════════════════════════════ #

class MinimaxAgent:
    """
    Minimax con poda Alpha-Beta y grabación completa del árbol de búsqueda.
    """

    def __init__(
        self,
        player_id: int,
        max_depth: int = 4,
        time_limit: float = 1.5,
        alpha_beta: bool = True,
        euclidean: bool = False,
    ) -> None:
        self.player_id  = player_id
        self.max_depth  = max_depth
        self.time_limit = time_limit
        self.alpha_beta = alpha_beta
        self.euclidean  = euclidean

        self.nodes_expanded:  int   = 0
        self.nodes_pruned:    int   = 0
        self.execution_time:  float = 0.0
        self._t0:             float = 0.0

        # Árbol de la última búsqueda (None hasta el primer movimiento)
        self.last_tree: Optional[MinimaxNode] = None
        self.last_evaluations: List[Tuple[Tuple[Tile, str], float]] = []

    # ------------------------------------------------------------------ #
    #  Interfaz pública                                                    #
    # ------------------------------------------------------------------ #

    def find_best_move(
        self,
        state: GameState,
        record_tree: bool = True,
    ) -> Optional[Tuple[Tile, str]]:
        """Calcula el mejor movimiento y (opcionalmente) graba el árbol."""
        self._t0            = time.perf_counter()
        self.nodes_expanded = 0
        self.nodes_pruned   = 0
        self.last_tree      = None
        self.last_evaluations = []

        moves = state.get_valid_moves()
        if not moves:
            self.execution_time = time.perf_counter() - self._t0
            return None
        if len(moves) == 1:
            self.execution_time = time.perf_counter() - self._t0
            if record_tree:
                t, s = moves[0]
                val, details = self._eval_with_details(state)
                node = MinimaxNode(
                    move=moves[0], score=val,
                    eval_details=details,
                    depth=0, is_max=True, is_terminal=True,
                    is_chosen=True,
                    label=f"{t}→{'L' if s == 'left' else 'R'}",
                )
                self.last_tree = node
            return moves[0]

        root, best_move = self._root_search(state, moves, record_tree)
        self.last_tree = root
        self.execution_time = time.perf_counter() - self._t0
        return best_move

    # ------------------------------------------------------------------ #
    #  Búsqueda raíz                                                       #
    # ------------------------------------------------------------------ #

    def _root_search(
        self,
        state: GameState,
        moves,
        record_tree: bool,
    ) -> Tuple[MinimaxNode, Optional[Tuple]]:

        root = MinimaxNode(
            move=None, score=0.0, depth=0, is_max=True,
            label="RAÍZ",
            alpha_in=-math.inf, beta_in=math.inf,
        )

        best_move:  Optional[Tuple] = None
        best_score: float           = -math.inf
        alpha = -math.inf
        beta  =  math.inf

        ordered = self._order_moves(state, moves, maximizing=True)

        for _, tile, side, nxt in ordered:
            label = f"{tile}→{'L' if side == 'left' else 'R'}"
            child = MinimaxNode(
                move=(tile, side), score=0.0, depth=1,
                is_max=False, label=label,
                alpha_in=alpha, beta_in=beta,
            ) if record_tree else None

            score = self._minimax(
                nxt, self.max_depth - 1, alpha, beta,
                maximizing=False,
                record=record_tree,
                parent=child,
            )

            if record_tree and child is not None:
                child.score = score
                root.children.append(child)

            self.last_evaluations.append(((tile, side), score))

            if score > best_score:
                best_score = score
                best_move  = (tile, side)

            if self.alpha_beta:
                alpha = max(alpha, score)

        root.score = best_score

        # ── Marcar ruta óptima ─────────────────────────────────────────── #
        if record_tree and best_move is not None:
            bt, bs = best_move
            root.is_chosen = True
            for child in root.children:
                cm = child.move
                if cm is not None and cm[0].canonical() == bt.canonical() and cm[1] == bs:
                    child.is_chosen = True
                    self._mark_path(child)
                    break

        return root, best_move

    # ------------------------------------------------------------------ #
    #  Minimax recursivo                                                   #
    # ------------------------------------------------------------------ #

    def _minimax(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        record: bool = True,
        parent: Optional[MinimaxNode] = None,
    ) -> float:
        self.nodes_expanded += 1

        if time.perf_counter() - self._t0 > self.time_limit:
            val = self._eval(state)
            if parent: parent.score = val; parent.is_terminal = True
            return val

        if depth == 0 or state.is_terminal():
            val, details = self._eval_with_details(state)
            if parent:
                parent.score = val
                parent.eval_details = details
                parent.is_terminal = True
            return val

        moves = state.get_valid_moves()
        if not moves:
            nxt = state.apply_pass()
            return self._minimax(nxt, depth - 1, alpha, beta,
                                 not maximizing, record, parent)

        ordered = self._order_moves(state, moves, maximizing)

        if maximizing:
            value  = -math.inf
            pruned = False
            for _, tile, side, nxt in ordered:
                if pruned:
                    # Nodo no explorado por poda
                    if record and parent is not None:
                        lbl  = f"{tile}→{'L' if side=='left' else 'R'}"
                        pd   = MinimaxNode(
                            move=(tile, side), score=0.0,
                            depth=parent.depth + 1, is_max=True,
                            label=lbl, was_pruned=True, is_terminal=True,
                            alpha_in=alpha, beta_in=beta,
                        )
                        parent.children.append(pd)
                    continue

                child = None
                if record and parent is not None:
                    lbl   = f"{tile}→{'L' if side=='left' else 'R'}"
                    child = MinimaxNode(
                        move=(tile, side), score=0.0,
                        depth=parent.depth + 1, is_max=True,
                        label=lbl, alpha_in=alpha, beta_in=beta,
                    )

                cv = self._minimax(nxt, depth - 1, alpha, beta,
                                   False, record, child)

                if record and parent is not None and child is not None:
                    child.score = cv
                    parent.children.append(child)

                if cv > value:
                    value = cv

                if self.alpha_beta:
                    alpha = max(alpha, value)
                    if value >= beta:
                        self.nodes_pruned += 1
                        pruned = True

            if parent: parent.score = value
            return value

        else:   # minimizando
            value  = math.inf
            pruned = False
            for _, tile, side, nxt in ordered:
                if pruned:
                    if record and parent is not None:
                        lbl = f"{tile}→{'L' if side=='left' else 'R'}"
                        pd  = MinimaxNode(
                            move=(tile, side), score=0.0,
                            depth=parent.depth + 1, is_max=False,
                            label=lbl, was_pruned=True, is_terminal=True,
                            alpha_in=alpha, beta_in=beta,
                        )
                        parent.children.append(pd)
                    continue

                child = None
                if record and parent is not None:
                    lbl   = f"{tile}→{'L' if side=='left' else 'R'}"
                    child = MinimaxNode(
                        move=(tile, side), score=0.0,
                        depth=parent.depth + 1, is_max=False,
                        label=lbl, alpha_in=alpha, beta_in=beta,
                    )

                cv = self._minimax(nxt, depth - 1, alpha, beta,
                                   True, record, child)

                if record and parent is not None and child is not None:
                    child.score = cv
                    parent.children.append(child)

                if cv < value:
                    value = cv

                if self.alpha_beta:
                    beta = min(beta, value)
                    if value <= alpha:
                        self.nodes_pruned += 1
                        pruned = True

            if parent: parent.score = value
            return value

    # ------------------------------------------------------------------ #
    #  Utilidades                                                          #
    # ------------------------------------------------------------------ #

    def _mark_path(self, node: MinimaxNode) -> None:
        """Marca recursivamente la ruta óptima descendiente."""
        if not node.children:
            return
        valid = [c for c in node.children if not c.was_pruned]
        if not valid:
            return
        best = (max if node.is_max else min)(valid, key=lambda c: c.score)
        best.is_chosen = True
        self._mark_path(best)

    def _order_moves(self, state, moves, maximizing):
        scored = []
        for tile, side in moves:
            nxt   = state.apply_move(tile, side)
            quick = self._eval(nxt)
            scored.append((quick, tile, side, nxt))
        scored.sort(key=lambda x: x[0], reverse=maximizing)
        return scored

    def _eval(self, state: GameState) -> float:
        val, _ = self._eval_with_details(state)
        return val

    def _eval_with_details(self, state: GameState) -> Tuple[float, Dict[str, float]]:
        winner = state.winner()
        if winner == self.player_id:
            return _WIN_SCORE, {"Estado": _WIN_SCORE}
        if winner is not None:
            return _LOSS_SCORE, {"Estado": _LOSS_SCORE}
        if state.is_terminal():
            return 0.0, {"Estado": 0.0}

        opp      = 1 - self.player_id
        my_hand  = state.hands[self.player_id]
        opp_hand = state.hands[opp]

        if state.board.is_empty():
            diff = Heuristics.hand_pip_value(opp_hand) - Heuristics.hand_pip_value(my_hand)
            val = _W_VALUE * diff
            return val, {"Pips (Δ)": val}

        le = state.board.left_end
        re = state.board.right_end

        delta_val  = Heuristics.hand_pip_value(opp_hand) - Heuristics.hand_pip_value(my_hand)
        delta_play = (Heuristics.playability_ratio(my_hand, le, re)
                      - Heuristics.playability_ratio(opp_hand, le, re)) * 10.0
        delta_dist = (Heuristics.hand_distance(opp_hand, le, re, euclidean=self.euclidean)
                      - Heuristics.hand_distance(my_hand,  le, re, euclidean=self.euclidean))
        delta_size = (len(opp_hand) - len(my_hand)) * 5.0

        c_val  = _W_VALUE * delta_val
        c_play = _W_PLAY  * delta_play
        c_dist = _W_DIST  * delta_dist
        c_size = _W_SIZE  * delta_size
        total  = c_val + c_play + c_dist + c_size

        details = {
            "h(n) Pips (Δ)": c_val,
            "Conectividad": c_play,
            "M. Dist (Ends)": c_dist,
            "Draft Size": c_size
        }

        return total, details
