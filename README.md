# Simulador de Torneo (32 equipos · 8 grupos · 4 niveles)

Simulador Monte Carlo de una competición con fase de grupos y, opcionalmente,
fase final eliminatoria. El objetivo es estudiar cómo cambian las
probabilidades de clasificación y de éxito de los equipos según el método de
sorteo de grupos (balanceado por bombos vs. completamente aleatorio).

## Modelo

### Equipos y niveles
- 32 equipos, 4 niveles, 8 equipos por nivel.
- Los niveles son únicamente etiquetas (1 = mejor, 4 = peor): toda la lógica
  consume el atributo `level`. Los nombres son cosméticos para evitar
  cualquier sesgo accidental.

### Probabilidades
- Centralizadas en `src/probabilities.py`.
- Una matriz simétrica entre niveles guarda P(victoria), P(empate) y
  P(derrota) para cada par (i, j).
- Coherencia validada en runtime:
  - P(victoria i,j) + P(empate i,j) + P(derrota i,j) = 1.
  - P(empate i,j) = P(empate j,i).
  - P(victoria i,j) = P(derrota j,i).
- Valores por defecto (modificables desde el dashboard):

  | Nivel A | Nivel B | P(victoria A) | P(empate) | P(victoria B) |
  |---------|---------|---------------|-----------|---------------|
  | 1       | 1       | 0.35          | 0.30      | 0.35          |
  | 1       | 2       | 0.55          | 0.25      | 0.20          |
  | 1       | 3       | 0.70          | 0.20      | 0.10          |
  | 1       | 4       | 0.82          | 0.13      | 0.05          |
  | 2       | 2       | 0.35          | 0.30      | 0.35          |
  | 2       | 3       | 0.55          | 0.25      | 0.20          |
  | 2       | 4       | 0.70          | 0.20      | 0.10          |
  | 3       | 3       | 0.35          | 0.30      | 0.35          |
  | 3       | 4       | 0.58          | 0.24      | 0.18          |
  | 4       | 4       | 0.35          | 0.30      | 0.35          |

### Fase de grupos
- 8 grupos de 4 equipos, round-robin de un solo turno (3 partidos por equipo).
- Puntuación: Victoria = 3, Empate = 1, Derrota = 0.
- Desempate dentro del grupo:
  1. Más puntos.
  2. Más victorias (sustituye a la diferencia de goles, ya que el modelo
     no simula marcadores).
  3. Desempate aleatorio reproducible (gestionado por la semilla).
- Clasifican los **2 primeros** de cada grupo (16 equipos).

### Fase final (opcional)
- Cuadro de octavos: 1º contra 2º. Se evita que dos equipos del mismo
  grupo se crucen en octavos reordenando aleatoriamente la lista de
  segundos.
- Eliminatorias sin empate: si el modelo base produciría empate, el
  resultado se condiciona excluyendo el empate, manteniendo la fuerza
  relativa entre niveles —`P(A gana | no empate) = p_w / (p_w + p_l)`.

### Sorteos comparados
- **Balanceado** (estilo cabezas de serie): los 8 equipos de **nivel 1** se
  reparten forzadamente, uno por grupo. Los 24 equipos restantes (niveles 2,
  3 y 4) se barajan **juntos** y se reparten 3 por grupo sin restricción de
  nivel. Es decir: solo se separan los mejores; el resto puede agruparse de
  cualquier manera.
- **Aleatorio**: los 32 equipos se barajan y se reparten en 8 grupos de 4
  sin restricción.

## Estructura

```
Probabilidades_torneo_futbito/
├── app.py                  # Dashboard Streamlit (solo presentación)
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── teams.py            # Modelo Team y factory de los 32 equipos
│   ├── probabilities.py    # Matriz de probabilidades + validación
│   ├── group_draw.py       # Sorteos balanced / random
│   ├── match_simulator.py  # Simulación de un partido (fase grupos / KO)
│   ├── group_stage.py      # Round-robin y ranking
│   ├── knockout_stage.py   # Cuadro eliminatorio
│   ├── monte_carlo.py      # Orquestador de N simulaciones
│   ├── metrics.py          # Agregaciones por equipo, nivel, comparación
│   ├── plots.py            # Gráficos Plotly (sin dependencias de UI)
│   ├── theory.py           # Modelo analítico cerrado del Teorema 2
│   └── theory_plots.py     # Gráficos de la sección teórica
│
└── tests/
    ├── test_probabilities.py
    ├── test_group_draw.py
    ├── test_group_stage.py
    ├── test_monte_carlo.py
    └── test_theory.py
```

## Instalación

Requiere Python 3.9+.

```bash
pip install -r requirements.txt
```

## Uso

### Dashboard

```bash
streamlit run app.py
```

El dashboard permite:
- Elegir nº de simulaciones y semilla.
- Elegir modo: solo grupos / grupos + fase final.
- Elegir sorteo: balanceado / aleatorio / comparar ambos.
- Editar manualmente la matriz de probabilidades por niveles.
- Ver tablas (por equipo y por nivel) y gráficos comparativos.
- **Sección teórica interactiva**: demostración visual del Teorema 2
  (sigmoide ajustable, punto de inflexión en P=½, casos donde las curvas se
  igualan vs. casos donde divergen). Ver sección «Demostración del Teorema 2».

### Desde consola

```python
from src.probabilities import ProbabilityMatrix
from src.monte_carlo import run_simulation
from src.metrics import per_team_metrics, per_level_metrics, compare_scenarios

matrix = ProbabilityMatrix.from_default()

df_b = run_simulation(n_sims=2000, draw_type="balanced",
                      matrix=matrix, include_knockout=True, seed=42)
df_r = run_simulation(n_sims=2000, draw_type="random",
                      matrix=matrix, include_knockout=True, seed=42)

team_b  = per_team_metrics(df_b,  include_knockout=True)
level_b = per_level_metrics(team_b)
team_r  = per_team_metrics(df_r,  include_knockout=True)
level_r = per_level_metrics(team_r)

print(compare_scenarios(level_b, level_r, "Balanceado", "Aleatorio").round(4))
```

## Tests

```bash
pytest -q
```

Cubren coherencia de probabilidades, generación correcta de equipos y
grupos, exactamente 2 clasificados por grupo, ausencia de duplicados,
**reproducibilidad bajo misma semilla** y, en `test_theory.py`, las cinco
propiedades del Teorema 2 (equivalencia con fuerzas iguales, equivalencia
con k=0, promedio teórico ½, signo de la diferencia para fuertes/débiles
y existencia de al menos un cruce).

## Demostración del Teorema 2 (sección teórica del dashboard)

El documento adjunto demuestra que, bajo cualquier mecanismo de sorteo en el
que clasifiquen exactamente *r* equipos de *m*, se cumple que
$\frac{1}{n} \sum_i P_i = r/m$. Y, además (Teorema 2), las dos curvas
$i \mapsto P^{ale}_i$  y  $i \mapsto P^{bal}_i$ coinciden punto a punto **si y
solo si todas las fuerzas son iguales**.

El módulo `src/theory.py` modela esto analíticamente con la sigmoide

$$P_i = \sigma\bigl(k\,(s_i - \mu_{rival,i})\bigr),\qquad \sigma(x) = \frac{1}{1+e^{-kx}}.$$

La segunda derivada $\sigma''(x) = k^2\,\sigma(1-\sigma)(1-2\sigma)$ cambia
de signo exactamente cuando $\sigma = \tfrac{1}{2}$: ese es el **punto de
inflexión**. Por la desigualdad de Jensen, ese cambio de curvatura es la
razón por la que un sorteo con varianza en la dificultad (aleatorio) y otro
sin varianza (balanceado) no pueden producir la misma curva, **salvo en dos
casos degenerados**: cuando las fuerzas son iguales (spread = 0) o cuando la
pendiente $k = 0$ (la curva colapsa a una recta horizontal en $\tfrac{1}{2}$
y el modelo se vuelve trivial).

El dashboard incluye sliders para `spread` y `k` que permiten transitar
entre estos casos visualmente.

### Limitaciones del modelo teórico

El módulo `theory.py` agrega la fuerza rival ANTES de la sigmoide
($\sigma(s_i - \overline{s}_{rival})$). La simulación Monte Carlo real
aplica la sigmoide partido a partido, lo que activa Jensen al máximo. Por
tanto, el modelo cerrado **subestima la magnitud** (pero no el signo) de
las diferencias entre sorteos. Es correcto como demostración cualitativa
del Teorema 2.

## Cómo interpretar los resultados

- `prob_qualified`: probabilidad de que el equipo termine 1º o 2º del grupo.
- `prob_at_least_one_win`: P(ganar ≥1 partido en fase de grupos).
- `prob_win_and_draw`: P(ganar ≥1 partido **y** empatar ≥1 partido).
- `avg_points` / `std_points`: media y std de puntos del equipo a lo largo
  de las simulaciones.
- `avg_position`: posición media (1 = primero, 4 = cuarto).
- `prob_reach_*`: P(alcanzar o pasar) cada ronda de la fase final.

A nivel de **nivel**, las columnas `_mean` son la media entre equipos del
mismo nivel y las columnas `_std` miden la dispersión **entre equipos**
(equipos del mismo nivel deberían mostrar valores muy parecidos: una std
alta sugiere insuficiencia de simulaciones).

Las columnas `*_diff_abs` y `*_diff_rel` de `compare_scenarios` indican
cuánto cambia cada métrica al pasar de un escenario a otro.

## Decisiones de diseño

1. **Sin simulación de goles** en el modelo base. Mantiene el modelo
   simple e interpretable: el desempate por victorias actúa como sustituto
   suficiente de la diferencia de goles.
2. **Una `np.random.Generator` por simulación**, derivada de la semilla
   maestra. Garantiza reproducibilidad total e independencia entre runs.
3. **Equipos genéricos por nombre** (`T1A…T4H`). Por construcción, no
   puede haber sesgo por nombre o posición; cualquier asimetría sería un
   bug, no una propiedad del modelo.
4. **Matriz de probabilidades simétrica con un único setter** que actualiza
   la entrada espejo automáticamente. La validación lanza `ValueError` con
   mensaje legible si la coherencia se rompe.
5. **Eliminatoria condicionada en lugar de un parámetro nuevo**:
   `P(gana A | no empate)` reusa el modelo base. No introducimos
   probabilidades adicionales que el usuario tendría que ajustar.
6. **Capa de presentación delgada**: `app.py` solo orquesta y renderiza.
   La lógica vive íntegramente en `src/`, lo que facilita testear y
   ejecutar el proyecto desde consola sin Streamlit.
7. **DataFrame "long"** como salida de Monte Carlo: una fila por
   (simulación, equipo). Permite calcular cualquier métrica posterior con
   pandas sin reejecutar.

## Limitaciones

- **No se simulan goles**: la diferencia de goles real no está reflejada.
  Si dos equipos quedan empatados a puntos y victorias, el desempate es
  aleatorio (reproducible).
- **Probabilidades estacionarias**: no se modela fatiga, lesiones,
  efecto local, ni momentum. Cada partido es independiente.
- **Bracket KO fijo tras octavos**: no se hace re-sorteo en cuartos o
  semifinales. Como los grupos son intercambiables, esto no afecta a las
  métricas agregadas, pero sí limita el análisis del cuadro.
- **Coherencia entre pares aristóteles**: la matriz solo garantiza
  consistencia local (i,j ↔ j,i). No se imponen restricciones globales
  tipo transitividad de fuerza, así que el usuario puede definir
  configuraciones físicamente extrañas (p.ej. nivel 1 < nivel 4) si lo
  desea explícitamente.

## Posibles ampliaciones

- Simulación de goles con un modelo Poisson por nivel.
- Cabezas de serie con re-sorteo por ronda en la fase final.
- Distribución empírica de campeones por equipo (no solo por nivel).
- Comparación con un tercer escenario (p.ej. anti-balanceado: 4 niveles 1
  juntos).
- Exportación de resultados a CSV/Excel desde el dashboard.

## Reproducibilidad

Toda la aleatoriedad se canaliza a través de `numpy.random.Generator` con
semilla configurable. Misma semilla ⇒ resultados idénticos. Esto está
verificado por test (`test_simulation_reproducibility_*`).
