"""
main.py — Punto de entrada del prototipo Dominó IA.

Ejecutar:
    python main.py

Dependencias:
    pip install pygame numpy
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

try:
    import pygame
except ImportError:
    print("ERROR: pygame no encontrado.\nInstálalo con:  pip install pygame")
    sys.exit(1)

from game.game_engine import DominoEngine
from ai.agent import DominoAgent, Strategy
from ui.game_ui import MenuScreen, GameScreen


def _build_agents(config: dict) -> dict:
    strategy = config["strategy"]
    depth    = config.get("depth", 4)
    if config["mode"] == "human_vs_ai":
        return {1: DominoAgent(1, strategy=strategy, minimax_depth=depth, name="IA")}
    return {
        0: DominoAgent(0, strategy=strategy, minimax_depth=depth, name="IA-1"),
        1: DominoAgent(1, strategy=strategy, minimax_depth=depth, name="IA-2"),
    }


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Domino IA | A* + Minimax")
    screen = pygame.display.set_mode((1200, 750), pygame.RESIZABLE)
    pygame.display.set_icon(_make_icon())

    menu = MenuScreen(screen)

    while True:
        config = menu.run()
        if config is None:
            break

        state  = DominoEngine.new_game()
        agents = _build_agents(config)

        try:
            game   = GameScreen(screen, state, agents, config)
            result = game.run()
        except SystemExit:
            break

        print("\n" + "=" * 60)
        print(f"  Partida finalizada  |  Modo: {config['mode']}")
        for pid, agent in agents.items():
            agent.metrics.print_report(agent.name)

        if result == "quit":
            break

    pygame.quit()
    sys.exit(0)


def _make_icon() -> pygame.Surface:
    icon = pygame.Surface((32, 32), pygame.SRCALPHA)
    icon.fill((0, 0, 0, 0))
    pygame.draw.rect(icon, (252, 248, 220), (2, 8, 28, 16), border_radius=3)
    pygame.draw.line(icon, (80, 80, 80), (16, 10), (16, 22), 2)
    for pos in [(8, 14), (24, 14), (8, 20), (24, 20)]:
        pygame.draw.circle(icon, (30, 30, 30), pos, 2)
    return icon


if __name__ == "__main__":
    main()
