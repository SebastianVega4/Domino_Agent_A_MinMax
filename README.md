# 🁢 Domino AI — Agente Inteligente con A* y Minimax

Prototipo de agente de IA competitivo para **Dominó doble-6 (2 jugadores)**,
desarrollado con **A\* Search**, **Minimax con poda α-β** y heurísticas de
**distancia Manhattan / Euclidiana**.

---

## Estructura del proyecto

```
domino_agent/
├── main.py                  # Punto de entrada (pygame)
├── requirements.txt
│
├── game/                    # Lógica del juego
│   ├── tile.py              # Ficha: punto 2D (a,b) en espacio de pips
│   ├── board.py             # Cadena del tablero, generación de movimientos
│   ├── game_state.py        # Snapshot completo para búsqueda
│   └── game_engine.py       # Fábrica, reglas, primer jugador
│
├── ai/                      # Módulos de inteligencia artificial
│   ├── heuristics.py        # d_Manhattan, d_Euclidiana, playability_ratio
│   ├── astar.py             # A* con h(n) basado en distancias
│   ├── minimax.py           # Minimax + poda Alpha-Beta
│   └── agent.py             # DominoAgent: estrategia Híbrida/A*/Minimax
│
├── ui/
│   └── game_ui.py           # Interfaz pygame: Menú → Juego → Métricas
│
└── utils/
    └── metrics.py           # MoveRecord, MetricsTracker, print_report()
```

---

## Instalación y ejecución

```bash
pip install pygame numpy
python main.py
```

---

## Algoritmos implementados

### 1. A\* con heurística de distancia

**Formulación como problema de camino mínimo:**

| Componente | Definición |
|---|---|
| Estado     | `(left_end, right_end, frozenset(hand), depth)` |
| g(n)       | Fichas jugadas hasta el nodo *n* |
| h(n)       | `Σ tile_distance(t, left_end, right_end)` para *t* en mano |
| Objetivo   | Mano vacía |

**Distancias usadas en h(n):**

```
d_Manhattan(tile(a,b), end e)  =  |a − e| + |b − e|
d_Euclidiana(tile(a,b), end e) =  √[(a−e)² + (b−e)²]
```

Cada ficha se trata como el punto P=(a,b) en el espacio de pips.
El extremo del tablero se mapea al punto diagonal Q=(e,e).
Distancia 0 ⟹ ficha jugable de inmediato.

### 2. Minimax con poda Alpha-Beta

- **MAX**: agente actual — maximiza función de evaluación.
- **MIN**: oponente — minimiza función de evaluación.

**Función de evaluación (nodos hoja):**

```
eval(s) = 0.40 · Δ_pip_value
        + 0.30 · Δ_playability × 10
        + 0.20 · Δ_manhattan_sum
        + 0.10 · Δ_hand_size × 5
```

donde `Δ_X = X_oponente − X_agente` (valor positivo ⟹ ventaja para el agente).

Terminales: +1000 (victoria), −1000 (derrota).

**Ordenación de movimientos:** los hijos se ordenan por evaluación rápida
antes de la llamada recursiva, maximizando los cortes β y α.

### 3. Estrategia Híbrida (default)

```
hand_size > 4  →  A*      (fase apertura / planificación)
hand_size ≤ 4  →  Minimax (fase final / adversarial)
```

---

## Modos de juego

| Modo | Descripción |
|---|---|
| **Humano vs IA** | P0 es humano (clic en ficha + L/R o clic en extremo del tablero). P1 es el agente. |
| **IA vs IA**     | Partida automática con visualización. Velocidad configurable. |

---

## Controles (Humano vs IA)

| Acción | Tecla/Evento |
|---|---|
| Seleccionar ficha | Clic en la ficha |
| Jugar a la izquierda | `L` o clic en el extremo izquierdo del tablero |
| Jugar a la derecha  | `R` o clic en el extremo derecho del tablero |
| Robar/pasar         | Botón `ROBAR / PASAR` (aparece cuando no hay movimientos) |
| Desplazar tablero   | ← → o rueda del ratón |
| Volver al menú      | `ESC` |

---

## Métricas recopiladas (panel derecho + consola)

| Métrica | Fuente |
|---|---|
| Nodos expandidos A\* | `AStarSearch.nodes_expanded` |
| Nodos generados A\*  | `AStarSearch.nodes_generated` |
| Nodos Minimax        | `MinimaxAgent.nodes_expanded` |
| Nodos podados α-β    | `MinimaxAgent.nodes_pruned` |
| Tiempo de ejecución  | `perf_counter()` por llamada |
| Tasa de victorias    | `MetricsTracker.summary()` |
| Turnos por partida   | Conteo en el bucle principal |

---

## Resultados experimentales (20 partidas, profundidad 4)

| Configuración | W0 | W1 | T.prom | A\* ms | MM ms |
|---|---|---|---|---|---|
| A\* vs A\*       | 9  | 11 | 28.3 | 0.34 | — |
| MM vs MM         | 9  | 6  | 36.6 | — | 0.42 |
| Híbrido vs Híb.  | 9  | 11 | 30.1 | 0.40 | 0.16 |
| A\* vs MM        | 6  | 9  | 28.8 | 0.70 | 0.36 |
| Híbrido vs MM    | 6  | 9  | 28.1 | 0.83 | 0.25 |

*(N=15 partidas por configuración — datos para tablas del informe IEEE)*

---

## Notas de diseño

- El **conjunto doble-6** contiene 28 fichas; se reparten 7 por jugador, dejando 14 en el pozo.
- El primer jugador se determina por la ficha doble más alta (o ficha de mayor valor si no hay dobles).
- **Boneyard**: si no hay movimientos, el jugador roba una ficha. Si el pozo está vacío, pasa. Dos pases consecutivos = partida bloqueada (gana quien tenga menor suma de pips).
- La IA usa información completa del estado (ambas manos visibles en búsqueda). Esto es una simplificación estándar para prototipos; trabajo futuro: información parcial con muestreo estocástico.
