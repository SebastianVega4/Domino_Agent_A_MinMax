"""
game_ui.py — Interfaz visual para el Agente de Dominó IA (rediseñada).

Pantallas
---------
MenuScreen  : Selección de modo, estrategia, profundidad, velocidad.
GameScreen  : Juego en vivo con tablero, manos, panel de métricas y
              botón de árbol Minimax.
TreeViewer  : Visualizador del árbol (importado de ui.tree_viewer).

Diseño visual
-------------
• Fondo de tapete verde con ligero gradiente radial.
• Fichas con sombra, pips precisos y radios suaves.
• Panel derecho con métricas en tiempo real y barras de progreso.
• Cadena del tablero desplazable con indicadores de extremo.
• Animación de "pensando" para la IA.
• Botón "🌲 Ver Árbol MM" que abre el TreeViewer.
"""
from __future__ import annotations
import math
import threading
import time
from typing import Optional, Tuple, List, Dict

import pygame
import pygame.gfxdraw

from game.tile import Tile
from game.board import Board
from game.game_state import GameState
from game.game_engine import DominoEngine
from ai.agent import DominoAgent, Strategy
from ai.minimax import MinimaxNode
from ui.tree_viewer import TreeViewer

# ═══════════════════════════════════════════════════════════════════════════ #
#  Paleta de colores                                                          #
# ═══════════════════════════════════════════════════════════════════════════ #
C_BG          = (22,  38,  22)
C_FELT        = (28,  50,  28)
C_LANE        = (24,  45,  24)
C_TILE_FACE   = (252, 248, 230)
C_TILE_BACK   = (48,  82,  48)
C_TILE_BRD    = (60,  60,  55)
C_TILE_SHD    = (10,  20,  10)
C_PIP         = (28,  28,  24)
C_SELECTED    = (255, 215,   0)
C_HINT        = (100, 230, 110)
C_TEXT        = (228, 228, 210)
C_TEXT_DIM    = (130, 140, 120)
C_ACCENT      = (255, 230,  0)
C_PANEL       = (16,  26,  16)
C_PANEL_BRD   = (50,  78,  50)
C_BTN         = (38,  72,  38)
C_BTN_HOV     = (58, 100,  58)
C_BTN_ACT     = (255, 185,  45)
C_WIN         = (70, 210,  80)
C_LOSE        = (210,  65,  65)
C_P0          = (80, 155, 255)
C_P1          = (255, 110,  75)
C_TREE_BTN    = (30,  55, 100)
C_TREE_BTN_H  = (50,  85, 145)
C_TREE_BTN_B  = (80, 140, 255)

# ═══════════════════════════════════════════════════════════════════════════ #
#  Pip patterns                                                               #
# ═══════════════════════════════════════════════════════════════════════════ #
_PIPS: Dict[int, List[Tuple[float, float]]] = {
    0: [],
    1: [(0.50, 0.50)],
    2: [(0.28, 0.28), (0.72, 0.72)],
    3: [(0.28, 0.28), (0.50, 0.50), (0.72, 0.72)],
    4: [(0.28, 0.28), (0.72, 0.28), (0.28, 0.72), (0.72, 0.72)],
    5: [(0.28, 0.28), (0.72, 0.28), (0.50, 0.50), (0.28, 0.72), (0.72, 0.72)],
    6: [(0.22, 0.22), (0.78, 0.22), (0.22, 0.50), (0.78, 0.50), (0.22, 0.78), (0.78, 0.78)],
}


def draw_tile(
    surf: pygame.Surface,
    tile: Tile,
    x: int, y: int, w: int, h: int,
    selected: bool = False,
    hint: bool = False,
    face_up: bool = True,
    shadow: bool = True,
) -> pygame.Rect:
    """Dibuja una ficha de dominó en (x, y) con dimensiones (w×h)."""
    rect = pygame.Rect(x, y, w, h)

    # Sombra
    if shadow:
        sr = pygame.Rect(x + 2, y + 3, w, h)
        pygame.draw.rect(surf, C_TILE_SHD, sr, border_radius=5)

    if not face_up:
        pygame.draw.rect(surf, C_TILE_BACK, rect, border_radius=5)
        bw  = 3 if selected else 1
        bc  = C_SELECTED if selected else C_TILE_BRD
        pygame.draw.rect(surf, bc, rect, bw, border_radius=5)
        # Patrón de retícula en el reverso
        for i in range(3, w - 3, 8):
            pygame.draw.line(surf, (38, 68, 38), (x + i, y + 4), (x + i, y + h - 4), 1)
        return rect

    # Cara
    pygame.draw.rect(surf, C_TILE_FACE, rect, border_radius=5)

    bw = 3 if (selected or hint) else 2
    bc = C_SELECTED if selected else (C_HINT if hint else C_TILE_BRD)
    pygame.draw.rect(surf, bc, rect, bw, border_radius=5)

    # Línea divisoria central
    mx = x + w // 2
    pygame.draw.line(surf, C_TILE_BRD, (mx, y + 5), (mx, y + h - 5), 2)

    # Pips
    _draw_pips(surf, tile.left,  x,      y, w // 2, h)
    _draw_pips(surf, tile.right, x + w // 2, y, w // 2, h)

    return rect


def _draw_pips(surf, value, cx, cy, cw, ch):
    r = max(2, min(cw, ch) // 7)
    for px, py in _PIPS.get(value, []):
        dx = int(cx + px * cw)
        dy = int(cy + py * ch)
        pygame.draw.circle(surf, C_PIP, (dx, dy), r)


# ═══════════════════════════════════════════════════════════════════════════ #
#  Botón reutilizable                                                         #
# ═══════════════════════════════════════════════════════════════════════════ #

class Button:
    def __init__(self, rect, label, font, body=None, border=None, text_col=None):
        self.rect      = rect
        self.label     = label
        self.font      = font
        self.body      = body  or C_BTN
        self.border    = border or C_PANEL_BRD
        self.text_col  = text_col or C_TEXT
        self._hover    = False

    def update(self, mp):
        self._hover = self.rect.collidepoint(mp)

    def draw(self, surf):
        col = tuple(min(255, c + 20) for c in self.body) if self._hover else self.body
        pygame.draw.rect(surf, col,       self.rect, border_radius=6)
        pygame.draw.rect(surf, self.border, self.rect, 2, border_radius=6)
        t = self.font.render(self.label, True, self.text_col)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


# ═══════════════════════════════════════════════════════════════════════════ #
#  MenuScreen                                                                  #
# ═══════════════════════════════════════════════════════════════════════════ #

class MenuScreen:
    MODES      = ["Humano vs IA", "IA vs IA"]
    STRATEGIES = ["Solo A*", "Solo Minimax"]
    SPEEDS     = ["Lento (1.2s)", "Normal (0.5s)", "Rápido (0.15s)"]
    SPEED_VALS = [1.2, 0.5, 0.15]
    DEPTHS     = ["Depth 3", "Depth 4", "Depth 5"]
    DEPTH_VALS = [3, 4, 5]

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen  = screen
        self.fL = pygame.font.SysFont("Segoe UI", 38, bold=True)
        self.fM = pygame.font.SysFont("Segoe UI", 20)
        self.fS = pygame.font.SysFont("Segoe UI", 15)

        self.sel = {"mode": 0, "strategy": 0, "depth": 1, "speed": 1}

    def _selector(self, label, options, sel_key, y):
        W = self.screen.get_width()
        lbl = self.fS.render(label, True, C_TEXT_DIM)
        self.screen.blit(lbl, (60, y - 22))

        rects = {}
        total_w = sum(self.fM.size(o)[0] + 28 for o in options) + 12 * len(options)
        x = max(60, (W - total_w) // 2)
        for i, opt in enumerate(options):
            tw, _ = self.fM.size(opt)
            r = pygame.Rect(x, y, tw + 28, 38)
            is_sel = (self.sel[sel_key] == i)
            col  = C_BTN_ACT if is_sel else C_BTN
            bcol = C_ACCENT  if is_sel else C_PANEL_BRD
            pygame.draw.rect(self.screen, col, r, border_radius=7)
            pygame.draw.rect(self.screen, bcol, r, 2, border_radius=7)
            tc = (25, 25, 25) if is_sel else C_TEXT
            t = self.fM.render(opt, True, tc)
            self.screen.blit(t, t.get_rect(center=r.center))
            rects[i] = r
            x += tw + 40
        return rects

    def run(self) -> Optional[dict]:
        clock = pygame.time.Clock()
        while True:
            W, H = self.screen.get_size()
            mp = pygame.mouse.get_pos()
            self.screen.fill(C_BG)

            # Gradiente radial sutil
            for r in range(300, 0, -30):
                alpha = max(0, 12 - (300 - r) // 25)
                if alpha == 0: break
                s = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
                pygame.draw.circle(s, (50, 90, 50, alpha), (r, r), r)
                self.screen.blit(s, (W // 2 - r, H // 2 - r - 100))

            # Título
            title = self.fL.render("[::]  DOMINÓ — Agente IA  [::]", True, C_ACCENT)
            self.screen.blit(title, title.get_rect(centerx=W // 2, y=40))
            sub = self.fS.render(
                "A*  ·  Minimax α-β  ·  Heurísticas Manhattan & Euclidiana",
                True, C_TEXT_DIM,
            )
            self.screen.blit(sub, sub.get_rect(centerx=W // 2, y=92))
            pygame.draw.line(self.screen, C_PANEL_BRD, (80, 115), (W - 80, 115), 1)

            # Selectores
            mode_r  = self._selector("MODO DE JUEGO",  self.MODES,      "mode",     175)
            strat_r = self._selector("ESTRATEGIA IA",  self.STRATEGIES, "strategy", 258)
            depth_r = self._selector("PROFUNDIDAD MM", self.DEPTHS,     "depth",    340)
            speed_r = self._selector("VELOCIDAD IA",   self.SPEEDS,     "speed",    422)

            pygame.draw.line(self.screen, C_PANEL_BRD, (80, 478), (W - 80, 478), 1)

            # Info
            info = [
                "Doble-6  ·  28 fichas  ·  7 por jugador  ·  14 en boneyard",
                "A*: Planificacion de ruta optima  |  Minimax: Teoria de juegos (Adversario)",
                "Presiona  T  o usa el mouse para ver el Arbol/Plan de cada jugada",
            ]
            for i, ln in enumerate(info):
                t = self.fS.render(ln, True, C_TEXT_DIM)
                self.screen.blit(t, t.get_rect(centerx=W // 2, y=490 + i * 22))

            # Botón Iniciar
            bw, bh = 260, 54
            btn = pygame.Rect(W // 2 - bw // 2, 568, bw, bh)
            bhov = btn.collidepoint(mp)
            pygame.draw.rect(self.screen, C_BTN_HOV if bhov else C_BTN, btn, border_radius=10)
            pygame.draw.rect(self.screen, C_ACCENT, btn, 2, border_radius=10)
            bt = self.fM.render(">>>  Iniciar", True, C_ACCENT)
            self.screen.blit(bt, bt.get_rect(center=btn.center))

            q = self.fS.render("ESC  para salir", True, C_TEXT_DIM)
            self.screen.blit(q, q.get_rect(centerx=W // 2, y=H - 28))

            pygame.display.flip()
            clock.tick(60)

            for event in pygame.event.get():
                if event.type == pygame.QUIT: return None
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: return None
                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    pos = event.pos
                    for i, r in mode_r.items():
                        if r.collidepoint(pos): self.sel["mode"]     = i
                    for i, r in strat_r.items():
                        if r.collidepoint(pos): self.sel["strategy"] = i
                    for i, r in depth_r.items():
                        if r.collidepoint(pos): self.sel["depth"]    = i
                    for i, r in speed_r.items():
                        if r.collidepoint(pos): self.sel["speed"]    = i
                    if btn.collidepoint(pos):
                        sm = {0: Strategy.ASTAR, 1: Strategy.MINIMAX}
                        return {
                            "mode":     "human_vs_ai" if self.sel["mode"] == 0 else "ai_vs_ai",
                            "strategy": sm[self.sel["strategy"]],
                            "depth":    self.DEPTH_VALS[self.sel["depth"]],
                            "ai_delay": self.SPEED_VALS[self.sel["speed"]],
                        }


# ═══════════════════════════════════════════════════════════════════════════ #
#  GameScreen                                                                 #
# ═══════════════════════════════════════════════════════════════════════════ #

TW_BOARD = 70
TH_BOARD = 38
TW_HAND  = 86
TH_HAND  = 46
TW_TOP   = 62
TH_TOP   = 34
HEADER_H = 72
HAND_H   = 72
BOARD_H  = 100
PANEL_W  = 230
FOOTER_H = 52


class GameScreen:
    def __init__(self, screen, state: GameState, agents: dict, config: dict):
        self.screen  = screen
        self.state   = state
        self.agents  = agents
        self.config  = config
        self.mode    = config["mode"]
        self.ai_delay= config.get("ai_delay", 0.5)

        self.fL = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.fM = pygame.font.SysFont("Segoe UI", 17)
        self.fS = pygame.font.SysFont("Segoe UI", 13)

        # Estado de interacción humana
        self.selected_idx:  int  = -1
        self.selected_tile: Optional[Tile] = None
        self.sel_moves:     list = []
        self.status_msg:    str  = ""
        self.board_scroll:  int  = 0
        self.move_log:      List[str] = []
        self.tree_history:  Dict[int, Tuple[MinimaxNode, str]] = {} # move_idx -> (tree, agent_name)
        self.current_move_idx: int = 0

        # IA
        self._ai_thinking: bool = False
        self._ai_move:     Optional[Tuple] = None
        self._think_dots:  int  = 0
        self._think_timer: float= 0.0

        # IA Arbol/Plan
        self._last_ai_tree: Optional[MinimaxNode] = None
        self._last_ai_algo: str = ""
        self._last_ai_agent: str = ""

        # Rects de interacción
        self._hand_rects:       List[pygame.Rect] = []
        self._left_end_rect:    Optional[pygame.Rect] = None
        self._right_end_rect:   Optional[pygame.Rect] = None
        self._log_rects:        List[pygame.Rect] = []
        self._board_tile_rects: List[Tuple[pygame.Rect, int]] = [] # (rect, move_idx)
        self._scroll_l_rect:    Optional[pygame.Rect] = None
        self._scroll_r_rect:    Optional[pygame.Rect] = None

        # Botones
        self._btn_draw = Button(pygame.Rect(0, 0, 145, 38), "ROBAR / PASAR", self.fS)
        self._btn_menu = Button(pygame.Rect(12, 14, 105, 36), "<< Menu", self.fS)
        self._btn_tree = Button(
            pygame.Rect(0, 0, 158, 36), "[T] Ver Arbol MM", self.fS,
            body=C_TREE_BTN, border=C_TREE_BTN_B, text_col=(180, 210, 255),
        )
        self.viewer = TreeViewer(self.screen)
        
        # New: Flow controls
        self.auto_mode = True
        self._btn_next = Button(pygame.Rect(0, 0, 145, 36), "SIGUIENTE >>", self.fS, body=(100, 100, 50))
        self._btn_auto = Button(pygame.Rect(0, 0, 145, 36), "MODO: AUTO", self.fS)

        self._set_status()

    # ------------------------------------------------------------------ #
    #  Bucle principal                                                     #
    # ------------------------------------------------------------------ #

    def run(self) -> str:
        clock  = pygame.time.Clock()

        while True:
            W, H = self.screen.get_size()
            mp   = pygame.mouse.get_pos()
            dt   = clock.tick(60) / 1000.0

            self._update_think_anim(dt)
            self._update_ai()
            self._update_btn_positions(W, H)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "quit"
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return "menu"
                    if event.key in (pygame.K_LEFT, pygame.K_a):
                        self.board_scroll = max(0, self.board_scroll - TW_BOARD)
                    if event.key in (pygame.K_RIGHT, pygame.K_d):
                        self.board_scroll += TW_BOARD
                    if event.key == pygame.K_t:
                        # Ver árbol con T
                        self._open_tree(self.viewer)
                    if event.key == pygame.K_l:
                        self._try_play('left')
                    if event.key == pygame.K_r:
                        self._try_play('right')

                if event.type == pygame.MOUSEWHEEL:
                    self.board_scroll = max(0, self.board_scroll - event.y * (TW_BOARD * 2))

                if self._btn_menu.clicked(event):
                    return "menu"
                if self._btn_tree.clicked(event):
                    self._open_tree(self.viewer)
                if self._btn_auto.clicked(event):
                    self.auto_mode = not self.auto_mode
                    self._btn_auto.label = "MODO: AUTO" if self.auto_mode else "MODO: MANUAL"
                if not self.auto_mode and self._btn_next.clicked(event):
                    self._trigger_ai_move()

                self._handle_human(event)

            self._draw(W, H, mp)
            pygame.display.flip()

    # ------------------------------------------------------------------ #
    #  Árbol viewer                                                        #
    # ------------------------------------------------------------------ #

    def _open_tree(self, viewer: TreeViewer) -> None:
        title = f"{self._last_ai_agent} ({self._last_ai_algo})"
        viewer.set_tree(self._last_ai_tree, title)
        result = viewer.run()
        # Al volver, refrescar pantalla
        if result == "quit":
            # Propagar quit al caller
            raise SystemExit

    # ------------------------------------------------------------------ #
    #  IA                                                                  #
    # ------------------------------------------------------------------ #

    def _update_ai(self) -> None:
        if self.state.is_terminal():
            return
        cp = self.state.current_player
        if cp not in self.agents:
            return
        if self._ai_thinking:
            return

        if self._ai_move is not None:
            mv, score = self._ai_move
            self._ai_move = None
            if mv == ("PASS",):
                self._do_draw_pass()
            elif isinstance(mv, tuple) and len(mv) == 2:
                tile, side = mv
                self._apply_move(tile, side, ai=True, score=score)
            return

        # Only auto-trigger if in auto mode
        if self.auto_mode:
            self._trigger_ai_move()

    def _trigger_ai_move(self) -> None:
        cp = self.state.current_player
        if cp not in self.agents or self._ai_thinking:
            return
        
        self._ai_thinking = True
        agent = self.agents[cp]
        delay = self.ai_delay if self.auto_mode else 0.1

        def think():
            time.sleep(delay)
            moves = self.state.get_valid_moves()
            if moves:
                tile_side, score = agent.decide(self.state)
                # Capturar arbol y algoritmo
                self._last_ai_tree  = agent.last_tree
                self._last_ai_algo  = agent.last_algo_name
                self._last_ai_agent = agent.name
                
                if tile_side is None:
                    self._ai_move = (("PASS",), 0.0)
                else:
                    self._ai_move = (tile_side, score)
            else:
                self._ai_move = (("PASS",), 0.0)
            self._ai_thinking = False

        threading.Thread(target=think, daemon=True).start()

    def _update_think_anim(self, dt: float) -> None:
        self._think_timer += dt
        if self._think_timer > 0.4:
            self._think_timer = 0.0
            self._think_dots  = (self._think_dots + 1) % 4

    # ------------------------------------------------------------------ #
    #  Interacción humana                                                  #
    # ------------------------------------------------------------------ #

    def _handle_human(self, event: pygame.event.Event) -> None:
        if self.state.is_terminal():
            return
        if self.state.current_player in self.agents:
            return

        if self._btn_draw.clicked(event):
            self._do_draw_pass()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            pos = event.pos
            hand = self.state.hands[self.state.current_player]

            # Clic en ficha de mano
            for idx, r in enumerate(self._hand_rects):
                if r.collidepoint(pos):
                    if self.selected_idx == idx:
                        self.selected_idx   = -1
                        self.selected_tile  = None
                        self.sel_moves      = []
                    else:
                        self.selected_idx  = idx
                        self.selected_tile = hand[idx]
                        self.sel_moves = [
                            (t, s) for t, s in self.state.get_valid_moves()
                            if t.canonical() == hand[idx].canonical()
                        ]
                    return

            # Clic en extremo del tablero
            if self.selected_tile is not None:
                if self._left_end_rect and self._left_end_rect.collidepoint(pos):
                    self._try_play('left')
                    return
                if self._right_end_rect and self._right_end_rect.collidepoint(pos):
                    self._try_play('right')
                    return
            
            # Flechas de scroll
            if self._scroll_l_rect and self._scroll_l_rect.collidepoint(pos):
                self.board_scroll = max(0, self.board_scroll - TW_BOARD * 2)
                return
            if self._scroll_r_rect and self._scroll_r_rect.collidepoint(pos):
                self.board_scroll += TW_BOARD * 2
                return

            # Clic en Log para revisar arboles anteriores
            max_logs = 14
            start_idx = max(0, len(self.move_log) - max_logs)
            for i, r in enumerate(self._log_rects):
                move_idx = start_idx + i
                if r.collidepoint(pos) and move_idx in self.tree_history:
                    tree, name = self.tree_history[move_idx]
                    self.viewer.set_tree(tree, name)
                    self.viewer.run()
                    return
            
            # Clic en Fichas del Tablero para revisar historia
            for r, midx in self._board_tile_rects:
                if r.collidepoint(pos) and midx in self.tree_history:
                    tree, name = self.tree_history[midx]
                    self.viewer.set_tree(tree, name)
                    self.viewer.run()
                    return

    def _try_play(self, side: str) -> None:
        if self.selected_tile is None or self.state.current_player in self.agents:
            return
        for t, s in self.sel_moves:
            if s == side:
                self._apply_move(t, side)
                return

    def _apply_move(self, tile: Tile, side: str, ai: bool = False, score: Optional[float] = None) -> None:
        who = "IA" if ai else "Tu"
        scr = f" [{score:+.1f}]" if score is not None and abs(score) > 0.001 else ""
        self.move_log.append(f"{who}: {tile} -> {'L' if side=='left' else 'R'}{scr}")
        
        # Guardar historial de arboles/planes
        if self._last_ai_tree:
            self.tree_history[self.current_move_idx] = (self._last_ai_tree, f"{self._last_ai_agent} ({self._last_ai_algo})")
        
        self.current_move_idx += 1

        self.state = self.state.apply_move(tile, side)
        self.selected_idx   = -1
        self.selected_tile  = None
        self.sel_moves      = []
        # Auto-scroll al extremo derecho
        n = len(self.state.board.tiles)
        W = self.screen.get_width()
        lane = W - PANEL_W
        visible = lane - 40
        total_w = n * (TW_BOARD + 2) - 2
        self.board_scroll = max(0, total_w - visible)
        self._set_status()

    def _do_draw_pass(self) -> None:
        if self.state.boneyard:
            self.state = self.state.apply_draw()
            self.move_log.append("→ Roba ficha del boneyard")
        else:
            self.state = self.state.apply_pass()
            self.move_log.append("→ Pasa turno")
        self.selected_idx   = -1
        self.selected_tile  = None
        self.sel_moves      = []
        self._set_status()

    def _set_status(self) -> None:
        if self.state.is_terminal():
            w = self.state.winner()
            if w is None:
                self.status_msg = "EMPATE - Ambos bloqueados."
            elif self.mode == "human_vs_ai" and w not in self.agents:
                self.status_msg = "!!! GANASTE !!! (ESC para salir)"
            elif self.mode == "human_vs_ai":
                self.status_msg = "IA GANA (ESC para salir)"
            else:
                self.status_msg = f"GANADOR: JUGADOR {w + 1}"
        elif self._ai_thinking:
            dots = "." * self._think_dots
            self.status_msg = f"IA pensando{dots}"
        elif self.state.current_player in self.agents:
            self.status_msg = "IA calculando movimiento…"
        else:
            moves = self.state.get_valid_moves()
            if moves:
                self.status_msg = ("Tu turno — clic en ficha, luego L=izquierda / R=derecha "
                                   "o clic en extremo del tablero.")
            else:
                self.status_msg = "Sin movimientos válidos — pulsa ROBAR / PASAR."

    # ------------------------------------------------------------------ #
    #  Posición de botones                                                 #
    # ------------------------------------------------------------------ #

    def _update_btn_positions(self, W: int, H: int) -> None:
        board_y = HEADER_H + HAND_H + 8
        lane_w = W - PANEL_W
        
        # Botones en fila debajo del tablero
        self._btn_tree.rect = pygame.Rect(20, board_y + BOARD_H + 8, 150, 36)
        self._btn_auto.rect = pygame.Rect(180, board_y + BOARD_H + 8, 150, 36)
        self._btn_next.rect = pygame.Rect(340, board_y + BOARD_H + 8, 150, 36)
        
        # Botón draw/pass a la derecha de la fila
        self._btn_draw.rect = pygame.Rect(lane_w - 165, board_y + BOARD_H + 8, 145, 36)

    # ------------------------------------------------------------------ #
    #  Dibujo                                                              #
    # ------------------------------------------------------------------ #

    def _draw(self, W: int, H: int, mp: Tuple[int, int]) -> None:
        self.screen.fill(C_BG)
        self._draw_felt(W, H)
        self._draw_header(W)
        self._draw_top_hand(W)
        self._draw_board(W)
        self._draw_bottom_hand(W, H)
        self._draw_panel(W, H)
        self._draw_footer(W, H)
        self._draw_buttons(W, H, mp)
        if self.state.is_terminal():
            self._draw_overlay(W, H)

    def _draw_felt(self, W: int, H: int) -> None:
        """Fondo de tapete con gradiente radial sutil."""
        self.screen.fill(C_FELT, pygame.Rect(0, HEADER_H, W - PANEL_W, H - HEADER_H - FOOTER_H))

    # ── Header ─────────────────────────────────────────────────────────── #

    def _draw_header(self, W: int) -> None:
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, W, HEADER_H))
        pygame.draw.line(self.screen, C_PANEL_BRD, (0, HEADER_H), (W, HEADER_H), 1)

        # Título
        title = self.fL.render("Dominó IA", True, C_ACCENT)
        self.screen.blit(title, (16, 8))

        # Jugadores
        for pid, (lbl_base, col, ypos) in {
            0: ("J1", C_P0, 10),
            1: ("J2", C_P1, 36),
        }.items():
            lbl = "Tú" if (self.mode == "human_vs_ai" and pid not in self.agents) else (
                f"IA  [{self.agents[pid].strategy.value}]" if pid in self.agents else "Tú"
            )
            active = (self.state.current_player == pid)
            c      = col if active else C_TEXT_DIM
            marker = "> " if active else "  "
            txt = self.fS.render(f"{marker}{lbl_base}: {lbl}  |  fichas: {len(self.state.hands[pid])}", True, c)
            self.screen.blit(txt, (130, ypos + 10))

        # Boneyard
        by = self.fS.render(
            f"Boneyard: {len(self.state.boneyard)}  |  Tablero: {len(self.state.board)} fichas",
            True, C_TEXT_DIM,
        )
        self.screen.blit(by, by.get_rect(right=W - PANEL_W - 14, centery=HEADER_H // 2))

        # Hint de árbol
        hint = self.fS.render("T = Ver árbol Minimax", True, C_TREE_BTN_B)
        self.screen.blit(hint, hint.get_rect(right=W - PANEL_W - 14, y=HEADER_H - 18))

    # ── Mano superior ─────────────────────────────────────────────────── #

    def _draw_top_hand(self, W: int) -> None:
        hand = self.state.hands[1]
        face = (self.mode == "ai_vs_ai")
        y    = HEADER_H + 12
        lane = W - PANEL_W
        total= len(hand) * (TW_TOP + 4)
        x    = max(16, (lane - total) // 2)
        col  = C_P1 if self.state.current_player == 1 else C_TEXT_DIM
        lbl  = self.fS.render("IA-2" if 1 in self.agents else "Tú", True, col)
        self.screen.blit(lbl, (12, y + TH_TOP // 2 - 7))
        for tile in hand:
            draw_tile(self.screen, tile, x, y, TW_TOP, TH_TOP, face_up=face, shadow=False)
            x += TW_TOP + 4

    # ── Tablero ───────────────────────────────────────────────────────── #

    def _draw_board(self, W: int) -> None:
        lane  = W - PANEL_W
        by    = HEADER_H + HAND_H + 8
        brect = pygame.Rect(0, by, lane, BOARD_H)
        pygame.draw.rect(self.screen, C_LANE, brect)
        pygame.draw.rect(self.screen, C_PANEL_BRD, brect, 1)

        self._left_end_rect  = None
        self._right_end_rect = None
        self._board_tile_rects = []

        if self.state.board.is_empty():
            msg = self.fM.render("Tablero vacio - juega la primera ficha", True, C_TEXT_DIM)
            self.screen.blit(msg, msg.get_rect(center=brect.center))
            return

        tiles   = self.state.board.tiles
        n       = len(tiles)
        gap     = 2
        total_w = n * (TW_BOARD + gap) - gap
        visible = lane - 40 # Mas margen
        
        # Ajustar scroll
        self.board_scroll = max(0, min(self.board_scroll, max(0, total_w - visible)))

        tile_y = by + (BOARD_H - TH_BOARD) // 2
        clip_rect = pygame.Rect(20, by + 2, lane - 40, BOARD_H - 4)
        self.screen.set_clip(clip_rect)

        x0 = 20 - self.board_scroll
        for i, tile in enumerate(tiles):
            tx   = x0 + i * (TW_BOARD + gap)
            is_l = (i == 0)
            is_r = (i == n - 1)
            
            # Resaltar extremos si hay ficha seleccionada
            hint_l = (self.selected_tile is not None and is_l
                      and any(s == 'left'  for _, s in self.sel_moves))
            hint_r = (self.selected_tile is not None and is_r
                      and any(s == 'right' for _, s in self.sel_moves))
            
            r = draw_tile(self.screen, tile, tx, tile_y, TW_BOARD, TH_BOARD,
                          hint=(hint_l or hint_r), shadow=True)
            
            # Hitbox para historia (ajustada por el scroll)
            # Solo guardamos el rect si esta dentro del clip visualmente? Pygame se encarga del dibujo, nosotros de la colision.
            # Convertimos r a coordenadas absolutas si draw_tile devolviera relativo, pero devuelve absoluto.
            if is_l: self._left_end_rect  = r
            if is_r: self._right_end_rect = r
            
            # Mapear ficha i -> move_idx i
            # (En domino el tablero crece, la ficha i en la lista corresponde al movimiento i)
            self._board_tile_rects.append((r, i))

        self.screen.set_clip(None)

        # Flechas de scroll
        self._scroll_l_rect = pygame.Rect(5, by + BOARD_H//2 - 15, 25, 30)
        self._scroll_r_rect = pygame.Rect(lane - 30, by + BOARD_H//2 - 15, 25, 30)

        # Dibujar flechas mas grandes y claras
        if self.board_scroll > 0:
            pygame.draw.polygon(self.screen, C_ACCENT,
                [(5, by + BOARD_H//2), (20, by + BOARD_H//2 - 12), (20, by + BOARD_H//2 + 12)])
        
        if total_w - self.board_scroll > visible:
            pygame.draw.polygon(self.screen, C_ACCENT,
                [(lane - 5, by + BOARD_H//2), (lane - 20, by + BOARD_H//2 - 12), (lane - 20, by + BOARD_H//2 + 12)])

        # Etiquetas de extremo sin caracteres especiales
        if self.state.board.left_end is not None:
            le = self.fS.render(f"<< {self.state.board.left_end}", True, C_ACCENT)
            self.screen.blit(le, (25, by + BOARD_H - 18))
        if self.state.board.right_end is not None:
            re = self.fS.render(f"{self.state.board.right_end} >>", True, C_ACCENT)
            self.screen.blit(re, (lane - 55, by + BOARD_H - 18))

    # ── Mano inferior ─────────────────────────────────────────────────── #

    def _draw_bottom_hand(self, W: int, H: int) -> None:
        hand     = self.state.hands[0]
        is_human = (0 not in self.agents)
        lane     = W - PANEL_W
        board_y  = HEADER_H + HAND_H + 8
        # Posición relativa al fondo
        y        = H - FOOTER_H - TH_HAND - 20 
        total    = len(hand) * (TW_HAND + 5)
        x        = max(12, (lane - total) // 2)
        col      = C_P0 if self.state.current_player == 0 else C_TEXT_DIM
        lbl      = self.fS.render("Tú" if is_human else "IA-1", True, col)
        self.screen.blit(lbl, (12, y + TH_HAND // 2 - 7))

        all_valid = {t.canonical() for t, _ in self.state.get_valid_moves()}
        self._hand_rects = []

        for i, tile in enumerate(hand):
            tx   = x + i * (TW_HAND + 5)
            sel  = (i == self.selected_idx)
            hint = is_human and (tile.canonical() in all_valid)
            face = is_human or (self.mode == "ai_vs_ai")
            # Levantar ficha seleccionada
            ty = y - 8 if sel else y
            r  = draw_tile(self.screen, tile, tx, ty, TW_HAND, TH_HAND,
                           selected=sel, hint=hint, face_up=face)
            self._hand_rects.append(r)

        # Instrucción bajo la mano
        if is_human and self.selected_tile is not None and self.sel_moves:
            sides = [s for _, s in self.sel_moves]
            parts = []
            if 'left'  in sides: parts.append("L ← izquierda")
            if 'right' in sides: parts.append("R → derecha")
            if parts:
                htxt = self.fS.render("  |  ".join(parts), True, C_HINT)
                self.screen.blit(htxt, htxt.get_rect(centerx=lane // 2, y=y + TH_HAND + 6))

    # ── Panel derecho ─────────────────────────────────────────────────── #

    def _draw_panel(self, W: int, H: int) -> None:
        px = W - PANEL_W
        pygame.draw.rect(self.screen, C_PANEL, (px, 0, PANEL_W, H))
        pygame.draw.line(self.screen, C_PANEL_BRD, (px, 0), (px, H), 1)

        y = 14

        def section(title: str) -> None:
            nonlocal y
            y += 6
            t = self.fS.render(title, True, C_ACCENT)
            self.screen.blit(t, (px + 10, y))
            pygame.draw.line(self.screen, C_PANEL_BRD,
                             (px + 10, y + 16), (W - 10, y + 16), 1)
            y += 24

        def row(label: str, value: str, vc=(200, 200, 180)) -> None:
            nonlocal y
            if y > H - 70: return
            l = self.fS.render(label, True, C_TEXT_DIM)
            v = self.fS.render(value,  True, vc)
            self.screen.blit(l, (px + 10, y))
            self.screen.blit(v, (W - v.get_width() - 10, y))
            y += 18

        def bar(value: float, max_val: float = 100.0, color=(80, 160, 80)) -> None:
            nonlocal y
            if y > H - 70: return
            bw = PANEL_W - 22
            bh = 6
            pygame.draw.rect(self.screen, (35, 50, 35), (px + 10, y, bw, bh), border_radius=3)
            filled = int(bw * min(value / max(max_val, 1), 1.0))
            if filled > 0:
                pygame.draw.rect(self.screen, color, (px + 10, y, filled, bh), border_radius=3)
            y += 12

        section("─ TABLERO ─")
        row("Fichas en tablero:", str(len(self.state.board)))
        row("Boneyard:",          str(len(self.state.boneyard)))
        row("Pases consecutivos:", str(self.state.pass_count))

        for pid, agent in self.agents.items():
            pname = agent.name
            section(f"--- {pname} ---")
            algo = agent.strategy.value
            row("Estrategia:", algo,
                (80, 140, 255) if "A*" in algo else (255, 110, 70))

            s = agent.summary()
            row("Movimientos:", str(s.get("total_moves_recorded", 0)))

            if s.get("astar_calls", 0) > 0:
                row("A* llamadas:", str(s["astar_calls"]))
                t_ms = s["astar_avg_time_ms"]
                row("A* t.prom (ms):", f"{t_ms:.2f}")
                row("A* nodos exp.:",  f"{s['astar_avg_nodes_expanded']:.0f}")
                bar(t_ms, 5.0, (80, 180, 255))

            if s.get("minimax_calls", 0) > 0:
                row("MM llamadas:", str(s["minimax_calls"]))
                t_ms = s["minimax_avg_time_ms"]
                row("MM t.prom (ms):", f"{t_ms:.2f}")
                row("MM nodos exp.:",  f"{s['minimax_avg_nodes']:.0f}")
                row("MM podados:",     f"{s['minimax_avg_pruned']:.0f}")
                bar(t_ms, 5.0, (255, 130, 70))

            # Arbol disponible
            if self._last_ai_tree is not None and self._last_ai_agent == agent.name:
                row("Plan/Arbol:", f"Dispon. ({self._last_ai_algo})", (120, 220, 120))
                
                evals = agent.last_evaluations
                if evals:
                    section("  └ CANDIDATOS:")
                    # Ordenar por el mejor score para mostrar arriba
                    sorted_evals = sorted(evals, key=lambda x: x[1], reverse=True)
                    for (tile, side), score in sorted_evals[:5]: # Mostrar max 5
                        lbl = f"{tile} → {'L' if side=='left' else 'R'}"
                        row(f"  {lbl}", f"{score:+.1f}")

        # Log
        section("─ MOVIMIENTOS ─")
        self._log_rects = []
        # Mostrar los últimos 14 y permitir click
        max_logs = 14
        start_idx = max(0, len(self.move_log) - max_logs)
        for i in range(start_idx, len(self.move_log)):
            entry = self.move_log[i]
            if y > H - 70: break
            
            has_tree = i in self.tree_history
            c = (255, 215, 100) if has_tree else C_P0 if (entry.startswith("IA") or entry.startswith("Tú")) else C_TEXT_DIM
            
            r = pygame.Rect(px + 10, y, PANEL_W - 20, 16)
            self._log_rects.append(r) # El índice en _log_rects coincidirá con el i si manejamos bien el click
            # Nota: Necesitamos que _handle_human sepa que el rect j corresponde al move_log start_idx + j
            
            if has_tree and r.collidepoint(pygame.mouse.get_pos()):
                pygame.draw.rect(self.screen, (40, 60, 40), r, border_radius=4)

            t = self.fS.render(entry[:30], True, c)
            self.screen.blit(t, (px + 10, y))
            y += 16

        if self.tree_history:
            y_hint = H - FOOTER_H - 18
            hint = self.fS.render("💡 Clic en log dorado para ver árbol", True, (100, 200, 100))
            self.screen.blit(hint, hint.get_rect(centerx=px + PANEL_W // 2, y=y_hint))

    # ── Footer ─────────────────────────────────────────────────────────── #

    def _draw_footer(self, W: int, H: int) -> None:
        fy = H - FOOTER_H
        pygame.draw.rect(self.screen, C_PANEL, (0, fy, W, FOOTER_H))
        pygame.draw.line(self.screen, C_PANEL_BRD, (0, fy), (W, fy), 1)
        col = C_ACCENT
        if self._ai_thinking:
            col = (150, 210, 255)
        msg = self.fS.render(self.status_msg, True, col)
        self.screen.blit(msg, msg.get_rect(centerx=(W - PANEL_W) // 2, centery=fy + FOOTER_H // 2))

    # ── Botones ────────────────────────────────────────────────────────── #

    def _draw_buttons(self, W: int, H: int, mp: Tuple[int, int]) -> None:
        self._btn_menu.update(mp)
        self._btn_menu.draw(self.screen)
        
        self._btn_tree.update(mp)
        self._btn_tree.draw(self.screen)
        
        self._btn_auto.update(mp)
        self._btn_auto.draw(self.screen)
        
        # Siguiente solo visible en manual y cuando es turno de IA
        if not self.auto_mode and self.state.current_player in self.agents:
            self._btn_next.update(mp)
            self._btn_next.draw(self.screen)

        # Robar/Pasar solo visible cuando es turno humano
        if self.state.current_player not in self.agents:
            self._btn_draw.update(mp)
            self._btn_draw.draw(self.screen)
            self._btn_draw.update(mp)
            self._btn_draw.draw(self.screen)

    # ── Overlay de fin de partida ──────────────────────────────────────── #

    def _draw_overlay(self, W: int, H: int) -> None:
        ow, oh = W - PANEL_W, 130
        oy     = H // 2 - oh // 2
        ov     = pygame.Surface((ow, oh), pygame.SRCALPHA)
        ov.fill((0, 0, 0, 170))
        self.screen.blit(ov, (0, oy))

        winner = self.state.winner()
        if winner is None:
            txt, col = "EMPATE", C_TEXT
        elif self.mode == "human_vs_ai" and winner not in self.agents:
            txt, col = "!!! GANASTE !!!", C_WIN
        elif self.mode == "human_vs_ai":
            txt, col = "IA GANA", C_LOSE
        else:
            txt = f"GANA JUGADOR {winner + 1}"
            col = C_P0 if winner == 0 else C_P1

        big = self.fL.render(txt, True, col)
        self.screen.blit(big, big.get_rect(centerx=ow // 2, y=oy + 18))

        detail = self.fS.render(
            f"Pip J1 = {self.state.hand_value(0)}    |    Pip J2 = {self.state.hand_value(1)}"
            f"    |    ESC = Menú    |    T = Ver árbol MM",
            True, C_TEXT_DIM,
        )
        self.screen.blit(detail, detail.get_rect(centerx=ow // 2, y=oy + 62))

        for pid, agent in self.agents.items():
            won = (winner == pid)
            agent.record_game_result(won, self.state.hand_value(pid), len(self.move_log))
