"""
tree_viewer.py — Visualizador interactivo del árbol Minimax.

Características
---------------
• Layout Reingold-Tilford simplificado (anchura de subárbol).
• Pan con arrastre del ratón (botón derecho o medio).
• Zoom con rueda del ratón, centrado en el cursor.
• Botón "Ajustar" para encuadrar el árbol completo.
• Hover sobre un nodo → tooltip con detalles completos.
• Colores:
    - Raíz              : gris claro
    - Nodo MAX (agente) : azul
    - Nodo MIN (oponente): naranja/rojo
    - Ruta elegida      : borde dorado grueso + fondo brillante
    - Podado α-β        : gris oscuro + tachado "✂"
    - Hoja WIN          : verde
    - Hoja LOSS         : rojo
• Aristas:
    - Ruta elegida      : dorado, grosor 3
    - Normal            : gris claro, grosor 1
    - Hacia podado      : gris oscuro, grosor 1, punteado simulado
• Leyenda fija en la esquina inferior izquierda.
• Contador de nodos, podados y profundidad en la cabecera.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple, List

import pygame

from ai.minimax import MinimaxNode

# ─── Paleta de colores ──────────────────────────────────────────────────── #
C_BG          = (12,  18,  12)
C_PANEL       = (18,  28,  18)
C_BORDER      = (40,  65,  40)
C_TEXT        = (240, 240, 240)   # Más claro
C_TEXT_DIM    = (160, 170, 160)   # Más claro
C_ACCENT      = (255, 230,  0)    # Amarillo más brillante y saturado

# Nodos
C_ROOT        = (70,  80,  70)    # Raíz
C_MAX_BODY    = (20,  50, 110)    # MAX → azul oscuro
C_MAX_BORDER  = (80, 140, 255)    # MAX → borde azul claro
C_MIN_BODY    = (100, 35,  20)    # MIN → rojo oscuro
C_MIN_BORDER  = (255, 110,  70)   # MIN → borde naranja
C_CHOSEN_GLOW = (255, 215,   0)   # Dorado — borde de ruta elegida
C_CHOSEN_BODY_MAX = (30, 75, 160) # MAX elegido — body más brillante
C_CHOSEN_BODY_MIN = (140, 50, 30) # MIN elegido
C_PRUNED_BODY = (35,  35,  35)
C_PRUNED_BRD  = (70,  70,  65)
C_WIN_BODY    = (20,  90,  40)
C_WIN_BRD     = (60, 220,  80)
C_LOSS_BODY   = (90,  20,  20)
C_LOSS_BRD    = (220,  60,  60)

# Aristas
C_EDGE_CHOSEN = (255, 215,   0)
C_EDGE_NORMAL = (80,  100,  80)
C_EDGE_PRUNED = (50,   55,  50)
C_EDGE_ROOT   = (120, 140, 120)

# ─── Dimensiones de nodo ────────────────────────────────────────────────── #
NODE_W   = 120
NODE_H   = 62
H_GAP    = 16    # Gap horizontal entre hermanos
V_GAP    = 72    # Gap vertical entre niveles
EDGE_W_CHOSEN = 3
EDGE_W_NORMAL = 1


class TreeViewer:
    """
    Pantalla de visualización del árbol Minimax.

    Uso
    ---
        viewer = TreeViewer(screen)
        viewer.set_tree(agent.last_mm_tree, agent_name="IA-1")
        result = viewer.run()   # 'menu', 'back', 'quit'
    """

    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen

        self.font_title = pygame.font.SysFont("Segoe UI", 22, bold=True)
        self.font_node  = pygame.font.SysFont("Segoe UI", 12, bold=True)
        self.font_small = pygame.font.SysFont("Segoe UI", 11)
        self.font_tip   = pygame.font.SysFont("Segoe UI", 13)
        self.font_leg   = pygame.font.SysFont("Segoe UI", 12)

        self.root:       Optional[MinimaxNode] = None
        self.agent_name: str = ""
        self._all_nodes: List[MinimaxNode] = []   # Flat list for hit-testing

        # Viewport
        self.pan_x:   float = 0.0
        self.pan_y:   float = 0.0
        self.zoom:    float = 1.0
        self._dragging = False
        self._drag_origin: Tuple[int, int] = (0, 0)
        self._pan_origin:  Tuple[float, float] = (0.0, 0.0)

        # Hover
        self._hovered: Optional[MinimaxNode] = None

        # Canvas virtual (en coordenadas de árbol, antes de zoom+pan)
        self._canvas_w: float = 0.0
        self._canvas_h: float = 0.0

    # ------------------------------------------------------------------ #
    #  API pública                                                         #
    # ------------------------------------------------------------------ #

    def set_tree(self, root: Optional[MinimaxNode], agent_name: str = "") -> None:
        self.root       = root
        self.agent_name = agent_name
        self._all_nodes = []
        if root is not None:
            self._compute_layout()
            self._collect_nodes(root)
            self._auto_fit()

    def run(self) -> str:
        """Bucle de eventos. Devuelve 'back' o 'quit'."""
        clock = pygame.time.Clock()

        while True:
            W, H = self.screen.get_size()
            mp = pygame.mouse.get_pos()

            for event in pygame.event.get():
                result = self._handle_event(event, W, H)
                if result:
                    return result

            self._update_hover(mp)
            self._draw(W, H, mp)
            pygame.display.flip()
            clock.tick(60)

    # ------------------------------------------------------------------ #
    #  Eventos                                                             #
    # ------------------------------------------------------------------ #

    def _handle_event(self, event: pygame.event.Event, W: int, H: int) -> Optional[str]:
        if event.type == pygame.QUIT:
            return "quit"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return "back"
            if event.key == pygame.K_f:
                self._auto_fit()
            if event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                self._zoom_at((W // 2, H // 2), 1.15)
            if event.key == pygame.K_MINUS:
                self._zoom_at((W // 2, H // 2), 1 / 1.15)

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button in (2, 3):   # Botón medio o derecho → pan
                self._dragging    = True
                self._drag_origin = event.pos
                self._pan_origin  = (self.pan_x, self.pan_y)
            if event.button == 4:
                self._zoom_at(event.pos, 1.12)
            if event.button == 5:
                self._zoom_at(event.pos, 1 / 1.12)

            # Botón izquierdo en el botón "Volver"
            if event.button == 1:
                btn = self._back_btn_rect(W, H)
                if btn.collidepoint(event.pos):
                    return "back"
                fit_btn = self._fit_btn_rect(W, H)
                if fit_btn.collidepoint(event.pos):
                    self._auto_fit()

        if event.type == pygame.MOUSEBUTTONUP:
            if event.button in (2, 3):
                self._dragging = False

        if event.type == pygame.MOUSEMOTION:
            if self._dragging:
                dx = event.pos[0] - self._drag_origin[0]
                dy = event.pos[1] - self._drag_origin[1]
                self.pan_x = self._pan_origin[0] + dx
                self.pan_y = self._pan_origin[1] + dy

        if event.type == pygame.MOUSEWHEEL:
            mp = pygame.mouse.get_pos()
            factor = 1.10 if event.y > 0 else 1 / 1.10
            self._zoom_at(mp, factor)

        return None

    # ------------------------------------------------------------------ #
    #  Layout                                                              #
    # ------------------------------------------------------------------ #

    @property
    def type_label(self) -> str:
        return "MAX [UP]" if self.is_max else "MIN [DOWN]"

    def _compute_layout(self) -> None:
        if self.root is None:
            return
        self._assign_widths(self.root)
        self._assign_positions(self.root, 0.0, 0)
        self._canvas_w = self.root._subtree_w
        self._canvas_h = self._tree_depth(self.root) * (NODE_H + V_GAP) + NODE_H

    def _assign_widths(self, node: MinimaxNode) -> float:
        """Cómputo bottom-up de la anchura del subárbol."""
        min_w = float(NODE_W + H_GAP)
        if not node.children:
            node._subtree_w = min_w
            return min_w
        total = sum(self._assign_widths(c) for c in node.children)
        node._subtree_w = max(min_w, total)
        return node._subtree_w

    def _assign_positions(self, node: MinimaxNode, x_start: float, depth: int) -> None:
        """Asignación top-down de posición de cada nodo."""
        node._x = x_start + node._subtree_w / 2 - NODE_W / 2
        node._y = float(depth * (NODE_H + V_GAP))

        child_x = x_start
        for child in node.children:
            self._assign_positions(child, child_x, depth + 1)
            child_x += child._subtree_w

    def _tree_depth(self, node: MinimaxNode) -> int:
        if not node.children:
            return 0
        return 1 + max(self._tree_depth(c) for c in node.children)

    def _collect_nodes(self, node: MinimaxNode) -> None:
        self._all_nodes.append(node)
        for child in node.children:
            self._collect_nodes(child)

    # ------------------------------------------------------------------ #
    #  Viewport                                                            #
    # ------------------------------------------------------------------ #

    def _auto_fit(self) -> None:
        """Encuadra todo el árbol centrado en la pantalla."""
        W, H = self.screen.get_size()
        if self._canvas_w == 0 or self._canvas_h == 0:
            return
        header_h = 70
        legend_h = 110
        avail_w  = W - 40
        avail_h  = H - header_h - legend_h - 20

        z_w = avail_w / max(self._canvas_w, 1)
        z_h = avail_h / max(self._canvas_h, 1)
        self.zoom  = min(z_w, z_h, 1.6)

        tree_pw  = self._canvas_w * self.zoom
        tree_ph  = self._canvas_h * self.zoom
        self.pan_x = (W - tree_pw) / 2
        self.pan_y = header_h + (avail_h - tree_ph) / 2

    def _zoom_at(self, pivot: Tuple[int, int], factor: float) -> None:
        """Zoom centrado en *pivot* (coordenadas de pantalla)."""
        px, py = pivot
        self.zoom  = max(0.15, min(self.zoom * factor, 5.0))
        # Ajustar pan para mantener el punto de la pantalla en el mismo lugar
        # (Se realiza tras el nuevo zoom)
        z  = self.zoom
        ox = (px - self.pan_x) / (z / factor)
        oy = (py - self.pan_y) / (z / factor)
        self.pan_x = px - ox * z
        self.pan_y = py - oy * z

    # ─── Conversión de coordenadas ─────────────────────────────────────── #

    def _to_screen(self, x: float, y: float) -> Tuple[int, int]:
        return (int(x * self.zoom + self.pan_x),
                int(y * self.zoom + self.pan_y))

    def _node_screen_rect(self, node: MinimaxNode) -> pygame.Rect:
        sx, sy = self._to_screen(node._x, node._y)
        return pygame.Rect(sx, sy, int(NODE_W * self.zoom), int(NODE_H * self.zoom))

    # ------------------------------------------------------------------ #
    #  Hover                                                               #
    # ------------------------------------------------------------------ #

    def _update_hover(self, mp: Tuple[int, int]) -> None:
        self._hovered = None
        for node in self._all_nodes:
            if self._node_screen_rect(node).collidepoint(mp):
                self._hovered = node
                break

    # ------------------------------------------------------------------ #
    #  Dibujo principal                                                    #
    # ------------------------------------------------------------------ #

    def _draw(self, W: int, H: int, mp: Tuple[int, int]) -> None:
        self.screen.fill(C_BG)
        self._draw_header(W)
        self._draw_legend(W, H)
        self._draw_buttons(W, H, mp)

        if self.root is None:
            msg = self.font_title.render(
                "No hay árbol disponible. Juega una partida primero.", True, C_TEXT_DIM
            )
            self.screen.blit(msg, msg.get_rect(center=(W // 2, H // 2)))
            return

        # Área de dibujo con clip para no solapar el header/legend
        clip = pygame.Rect(0, 65, W, H - 170)
        self.screen.set_clip(clip)
        self._draw_level_indicators(W)
        self._draw_edges(self.root)
        self._draw_nodes(self.root)
        self.screen.set_clip(None)

        if self._hovered is not None:
            self._draw_tooltip(self._hovered, W, H)

    def _draw_level_indicators(self, W: int) -> None:
        """Dibuja franjas de fondo y etiquetas para cada nivel de profundidad."""
        if self.root is None:
            return
        
        depth = self._tree_depth(self.root) + 1
        for d in range(depth):
            # Calcular banda en coordenadas de pantalla
            _, y0 = self._to_screen(0, d * (NODE_H + V_GAP) - V_GAP // 2)
            _, y1 = self._to_screen(0, d * (NODE_H + V_GAP) + NODE_H + V_GAP // 2)
            
            # Solo dibujar si es visible en el clip (65 a H-170)
            if y1 < 65 or y0 > self.screen.get_height() - 170:
                continue
            
            # Franja de fondo
            alpha = 15 if d % 2 == 0 else 5
            band_surf = pygame.Surface((W, y1 - y0), pygame.SRCALPHA)
            band_surf.fill((255, 255, 255, alpha))
            self.screen.blit(band_surf, (0, y0))
            
            # Línea divisoria
            pygame.draw.line(self.screen, (40, 60, 40), (0, y0), (W, y0), 1)
            
            # Etiqueta de capa
            is_max = (d % 2 == 0) # Raiz es depth 0, es MAX
            lbl_txt = f"NIVEL {d}: {'CAPA MAX (Agente)' if is_max else 'CAPA MIN (Rival)'}"
            lbl_col = C_MAX_BORDER if is_max else C_MIN_BORDER
            
            sur = self.font_small.render(lbl_txt, True, lbl_col)
            # Fondo para el texto
            pygame.draw.rect(self.screen, C_BG, (5, y0 + 5, sur.get_width() + 10, sur.get_height() + 4), border_radius=4)
            self.screen.blit(sur, (10, y0 + 7))

    # ── Header ─────────────────────────────────────────────────────────── #

    def _draw_header(self, W: int) -> None:
        pygame.draw.rect(self.screen, C_PANEL, (0, 0, W, 62))
        pygame.draw.line(self.screen, C_BORDER, (0, 62), (W, 62), 1)

        title = self.font_title.render(
            f"[T]  ARBOL MINIMAX  -  {self.agent_name}", True, C_ACCENT
        )
        self.screen.blit(title, (16, 10))

        if self.root is not None:
            total  = len(self._all_nodes)
            pruned = sum(1 for n in self._all_nodes if n.was_pruned)
            depth  = self._tree_depth(self.root)
            stats  = (f"Exp: {total - pruned}   |   "
                      f"Poda Alpha-Beta: {pruned}   |   "
                      f"Depth: {depth}   |   "
                      f"Raiz: {self.root.score_str}")
            st = self.font_small.render(stats, True, C_TEXT_DIM)
            self.screen.blit(st, (16, 40))

        hint = self.font_small.render(
            "Wheel: Zoom  |  Drag (R-Click): Pan  |  F: Fit  |  ESC: Back",
            True, C_TEXT_DIM,
        )
        self.screen.blit(hint, hint.get_rect(right=W - 16, y=42))

    # ── Leyenda ────────────────────────────────────────────────────────── #

    def _draw_legend(self, W: int, H: int) -> None:
        lh = 100
        ly = H - lh
        pygame.draw.rect(self.screen, C_PANEL, (0, ly, W, lh))
        pygame.draw.line(self.screen, C_BORDER, (0, ly), (W, ly), 1)

        items = [
            (C_MAX_BODY,   C_MAX_BORDER,   "MAX [^] - Capa Agente (Maximiza)"),
            (C_MIN_BODY,   C_MIN_BORDER,   "MIN [v] - Capa Rival (Minimiza)"),
            (C_WIN_BODY,   C_WIN_BRD,      "WIN - Estado Ganador"),
            (C_LOSS_BODY,  C_LOSS_BRD,     "LOSS - Estado Perdedor"),
            (C_PRUNED_BODY,C_PRUNED_BRD,   "[X] - Rama podada Alpha-Beta"),
        ]
        x = 16
        y = ly + 10
        for body, brd, txt in items:
            pygame.draw.rect(self.screen, body, (x, y, 22, 14), border_radius=3)
            pygame.draw.rect(self.screen, brd,  (x, y, 22, 14), 2, border_radius=3)
            lbl = self.font_leg.render(txt, True, C_TEXT_DIM)
            self.screen.blit(lbl, (x + 26, y))
            x += lbl.get_width() + 52
            if x > W - 200:
                x  = 16
                y += 20

        # Leyenda de aristas
        y2 = ly + 50
        # Dorada
        pygame.draw.line(self.screen, C_EDGE_CHOSEN, (16, y2 + 7), (50, y2 + 7), 3)
        lbl = self.font_leg.render("Ruta óptima elegida", True, C_ACCENT)
        self.screen.blit(lbl, (54, y2))
        # Gris normal
        pygame.draw.line(self.screen, C_EDGE_NORMAL, (220, y2 + 7), (254, y2 + 7), 1)
        lbl = self.font_leg.render("Rama explorada", True, C_TEXT_DIM)
        self.screen.blit(lbl, (258, y2))
        # Oscura (podada)
        for dx in range(0, 34, 8):
            pygame.draw.line(self.screen, C_EDGE_PRUNED,
                             (410 + dx, y2 + 7), (min(418 + dx, 444), y2 + 7), 1)
        lbl = self.font_leg.render("Rama podada (α-β)", True, C_TEXT_DIM)
        self.screen.blit(lbl, (448, y2))
        # Borde dorado elegido
        pygame.draw.rect(self.screen, C_CHOSEN_GLOW, (620, y2, 22, 14), 2, border_radius=3)
        lbl = self.font_leg.render("Borde dorado = elegido", True, C_ACCENT)
        self.screen.blit(lbl, (646, y2))

    # ── Botones ────────────────────────────────────────────────────────── #

    def _back_btn_rect(self, W: int, H: int) -> pygame.Rect:
        return pygame.Rect(W - 130, 12, 110, 38)

    def _fit_btn_rect(self, W: int, H: int) -> pygame.Rect:
        return pygame.Rect(W - 255, 12, 110, 38)

    def _draw_buttons(self, W: int, H: int, mp: Tuple[int, int]) -> None:
        for rect, label in [
            (self._back_btn_rect(W, H), "< Volver"),
            (self._fit_btn_rect(W, H),  "[F] Fit"),
        ]:
            hover = rect.collidepoint(mp)
            col   = (80, 120, 80) if hover else (40, 65, 40)
            pygame.draw.rect(self.screen, col,   rect, border_radius=6)
            pygame.draw.rect(self.screen, C_BORDER, rect, 1, border_radius=6)
            txt = self.font_tip.render(label, True, C_TEXT)
            self.screen.blit(txt, txt.get_rect(center=rect.center))

    # ── Aristas ────────────────────────────────────────────────────────── #

    def _draw_edges(self, node: MinimaxNode) -> None:
        cx = int(node._x * self.zoom + self.pan_x + NODE_W * self.zoom / 2)
        cy = int(node._y * self.zoom + self.pan_y + NODE_H * self.zoom)

        for child in node.children:
            ccx = int(child._x * self.zoom + self.pan_x + NODE_W * self.zoom / 2)
            ccy = int(child._y * self.zoom + self.pan_y)

            if child.is_chosen:
                color = C_EDGE_CHOSEN
                width = max(1, int(EDGE_W_CHOSEN * self.zoom * 0.8))
            elif child.was_pruned:
                color = C_EDGE_PRUNED
                width = 1
            else:
                color = C_EDGE_NORMAL
                width = 1

            if child.was_pruned:
                # Arista punteada simulada
                steps = max(6, int(abs(ccy - cy) / 8))
                for i in range(0, steps, 2):
                    t0 = i / steps
                    t1 = min((i + 1) / steps, 1.0)
                    px0 = int(cx + (ccx - cx) * t0)
                    py0 = int(cy + (ccy - cy) * t0)
                    px1 = int(cx + (ccx - cx) * t1)
                    py1 = int(cy + (ccy - cy) * t1)
                    pygame.draw.line(self.screen, color, (px0, py0), (px1, py1), width)
            else:
                pygame.draw.line(self.screen, color, (cx, cy), (ccx, ccy), width)

            self._draw_edges(child)

    # ── Nodos ──────────────────────────────────────────────────────────── #

    def _draw_nodes(self, node: MinimaxNode) -> None:
        self._render_node(node)
        for child in node.children:
            self._draw_nodes(child)

    def _render_node(self, node: MinimaxNode) -> None:
        rect = self._node_screen_rect(node)

        # ── Seleccionar colores ─────────────────────────────────────────── #
        if node.was_pruned:
            body, border = C_PRUNED_BODY, C_PRUNED_BRD
        elif node.label == "RAÍZ":
            body   = C_ROOT
            border = (110, 130, 110)
        elif node.score >= 999:
            body, border = C_WIN_BODY, C_WIN_BRD
        elif node.score <= -999:
            body, border = C_LOSS_BODY, C_LOSS_BRD
        elif node.is_max:
            body   = C_CHOSEN_BODY_MAX if node.is_chosen else C_MAX_BODY
            border = C_MAX_BORDER
        else:
            body   = C_CHOSEN_BODY_MIN if node.is_chosen else C_MIN_BODY
            border = C_MIN_BORDER

        # ── Dibuja cuerpo ──────────────────────────────────────────────── #
        # Sombra sutil
        shadow_rect = rect.move(2, 2)
        pygame.draw.rect(self.screen, (5, 10, 5, 160), shadow_rect, border_radius=6)
        
        pygame.draw.rect(self.screen, body,   rect, border_radius=6)

        # Borde elegido → dorado grueso con glow
        if node.is_chosen:
            # Glow exterior
            for i in range(1, 4):
                pygame.draw.rect(self.screen, (255, 215, 0, 30 // i), rect.inflate(i*2, i*2), border_radius=7)
            bw = max(2, int(3 * self.zoom))
            bc = C_CHOSEN_GLOW
        else:
            bw = max(1, int(self.zoom))
            bc = border
            
        pygame.draw.rect(self.screen, bc, rect, bw, border_radius=6)

        # ── Texto (solo si zoom suficiente) ───────────────────────────── #
        if self.zoom < 0.30:
            return

        inner_x = rect.x + 5
        inner_y = rect.y + 4
        line_h  = max(12, int(14 * self.zoom * 0.75))

        def txt(s: str, col, bold: bool = False) -> None:
            nonlocal inner_y
            if inner_y + line_h > rect.bottom - 2:
                return
            fn  = self.font_node if bold else self.font_small
            sur = fn.render(s, True, col)
            # Scale if zoomed
            if self.zoom != 1.0 and self.zoom > 0.4:
                nw = int(sur.get_width()  * min(self.zoom, 1.0))
                nh = int(sur.get_height() * min(self.zoom, 1.0))
                if nw > 0 and nh > 0:
                    sur = pygame.transform.scale(sur, (nw, nh))
            self.screen.blit(sur, (inner_x, inner_y))
            inner_y += line_h

        # Etiqueta principal (ficha/side o RAÍZ)
        lbl_col = (255, 255, 255) if node.is_chosen else C_TEXT
        txt(node.label, lbl_col, bold=True)

        # Puntuación
        sc_col = (
            (100, 255, 120) if node.score >= 999 else
            (255, 100, 100) if node.score <= -999 else
            (255, 255, 255) if node.is_chosen else
            C_TEXT
        )
        txt(f"Val: {node.score_str}", sc_col, bold=True)

        # h(n) resumido (si existe)
        h_val = node.eval_details.get("h(n) Pips (Δ)")
        if h_val is not None:
            txt(f"h(n): {h_val:+.1f}", C_TEXT_DIM)

        # Tipo MAX/MIN
        type_col = C_MAX_BORDER if node.is_max else C_MIN_BORDER
        if not node.was_pruned:
            txt(node.type_label, type_col)

        # Símbolo de podado
        if node.was_pruned:
            prune_txt = self.font_node.render("[X] PODA", True, (150, 80, 80))
            self.screen.blit(prune_txt, (inner_x, inner_y))

        # Hover glow
        if node is self._hovered:
            glow = pygame.Surface((rect.w, rect.h), pygame.SRCALPHA)
            glow.fill((255, 255, 255, 18))
            self.screen.blit(glow, rect.topleft)
            pygame.draw.rect(self.screen, (200, 200, 200), rect, 1, border_radius=6)

    # ── Tooltip ────────────────────────────────────────────────────────── #

    def _draw_tooltip(self, node: MinimaxNode, W: int, H: int) -> None:
        lines = [
            f"Movimiento   : {node.label}",
            f"Tipo         : {node.type_label} ({'Elije Max h' if node.is_max else 'Elije Min h'})",
            f"Puntuacion   : {node.score_str}",
            f"Profundidad  : {node.depth}",
        ]

        if node.eval_details:
            lines.append("--- Heuristica ---")
            for k, v in node.eval_details.items():
                lines.append(f"  {k:12}: {v:+.1f}")
        
        lines.extend([
            "--- Estado ---",
            f"Hoja         : {'Si' if node.is_terminal else 'No'}",
            f"Ruta optima  : {'[v] Elegida' if node.is_chosen else '---'}",
            f"Podado alpha-beta: {'[!] Si' if node.was_pruned else 'No'}",
            f"Hijos        : {len(node.children)}",
        ])
        pad = 10
        line_h = 18
        tw     = max(self.font_tip.size(l)[0] for l in lines) + pad * 2
        th     = len(lines) * line_h + pad * 2

        # Posicionar evitando bordes
        mx, my = pygame.mouse.get_pos()
        tx = mx + 16
        ty = my + 16
        if tx + tw > W: tx = mx - tw - 4
        if ty + th > H: ty = my - th - 4

        pygame.draw.rect(self.screen, (20, 30, 20), (tx, ty, tw, th), border_radius=6)
        pygame.draw.rect(self.screen, C_BORDER, (tx, ty, tw, th), 1, border_radius=6)

        for i, line in enumerate(lines):
            col = C_ACCENT if "elegida" in line and "✓" in line else C_TEXT
            s   = self.font_tip.render(line, True, col)
            self.screen.blit(s, (tx + pad, ty + pad + i * line_h))
