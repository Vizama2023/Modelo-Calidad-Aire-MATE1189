# Modelo-Calidad-Aire-MATE1189
Modelo matemático y simulación computacional de la dispersión espacial de contaminantes atmosféricos en una zona urbana. Este proyecto aplica herramientas de cálculo en varias variables (derivadas parciales, gradientes e integrales dobles) para analizar zonas de riesgo ambiental.

## 1. Contexto del problema

Calama, en la Región de Antofagasta, fue declarada oficialmente **zona saturada de contaminación atmosférica** en 2009, principalmente por arsénico, SO₂ y material particulado grueso (MP10). A solo ~15 km al norte de la ciudad se encuentra **Chuquicamata**, una de las minas de cobre a rajo abierto más grandes del mundo, operada por Codelco. El proceso de fundición del cobre libera MP10 al ambiente, y la ciudad convive con esa carga contaminante.

El propósito del proyecto es usar herramientas de cálculo multivariable(funciones de dos variables, gradiente, derivadas direccionales, linealización, puntos críticos, curvas de nivel e integrales dobles) para:

1. Modelar cómo se distribuye la concentración de MP10 en el espacio alrededor de las fuentes emisoras.
2. Calcular en qué dirección y con qué rapidez aumenta o disminuye esa concentración.
3. Determinar la cantidad total de contaminante y la concentración promedio en la región de estudio.
4. Comparar escenarios para entender qué tan sensible es el modelo a las condiciones reales.



### 1.1 Región de estudio

Se define un sistema de coordenadas local (en km) con origen en el centro de Calama, eje `x` apuntando al Este y eje `y` al Norte. La región de estudio es el rectángulo:

R = [-10, 10] x [-5, 25]   (en km)

Es decir, 20 km de ancho (Este-Oeste) por 30 km de alto (Norte-Sur), con área total `A(R) = 600 km²`. Esta región se eligió porque contiene tanto el centro urbano de Calama (en el origen) como las tres faenas mineras que se modelan como fuentes emisoras.

Esto es exactamente lo que hacen estas líneas:

python
x = np.linspace(-10, 10, 400)
y = np.linspace(-5, 25, 400)
X, Y = np.meshgrid(x, y)


`linspace` genera 400 puntos equiespaciados en cada eje (más puntos = malla más fina = gráficos más suaves y una integral numérica más precisa). `meshgrid` combina esos dos vectores en dos matrices `X` e `Y` de 400×400, de modo que cada par `(X[i,j], Y[i,j])` representa un punto físico de la ciudad. Esta malla es la que después se usa para evaluar la concentración en *todo* el territorio a la vez, en lugar de punto por punto.

### 1.2 Modelo de concentración para una fuente función gaussiana

La concentración generada por una única fuente emisora se modela como:

```
C(x, y) = A * exp( -[(x - xi)² + (y - yi)²] / k )
```

donde:

- **A** [µg/m³]: concentración máxima, alcanzada justo en el punto de emisión `(xi, yi)`.
- **k** [km²]: parámetro de dispersión. Controla qué tan rápido decae la concentración al alejarse de la fuente. A mayor `k`, la "nube" contaminante es más ancha y decae más lento (equivale, físicamente, a condiciones atmosféricas que dispersan menos el contaminante, como inversión térmica). A menor `k`, la concentración se concentra más cerca de la fuente.
- **(xi, yi)**: ubicación de la fuente, en km, dentro del sistema de coordenadas de Calama.


```python
def concentracion(x, y, A, k, xi, yi):
    return A * np.exp(-((x - xi) ** 2 + (y - yi) ** 2) / k)
```

`(x - xi)**2 + (y - yi)**2` es la distancia al cuadrado entre el punto evaluado y la fuente. Se usa la distancia *al cuadrado* (no la raíz) porque así queda naturalmente el exponente de una gaussiana; calcular la raíz sería más costoso computacionalmente y no aporta nada distinto.

Gracias a que `np.exp` y las operaciones `**`, `-`, `/` están vectorizadas en NumPy, esta misma función sirve tanto para evaluar la concentración en un solo punto (números `x`, `y`) como en toda la malla `X`, `Y` de una vez — no hace falta escribir un `for` recorriendo cada punto.

### 1.3 Extensión a múltiples fuentes (superposición lineal)

La ciudad no está afectada por una sola faena, sino por varias. El modelo asume **superposición lineal**: se suma la contribución de cada fuente, como si los contaminantes no reaccionaran químicamente entre sí:

```
C_total(x, y) = Σ Ai * exp( -[(x - xi)² + (y - yi)²] / ki )
```


```python
def campo_total(X, Y, fuentes):
    C = np.zeros_like(X)
    for (xi, yi, A, k) in fuentes:
        C += concentracion(X, Y, A, k, xi, yi)
    return C
```

`np.zeros_like(X)` crea una matriz de ceros del mismo tamaño que `X` (400×400), que actúa como "acumulador". El `for` recorre la lista `fuentes` y va sumando la nube gaussiana de cada una sobre la misma malla. El resultado es el campo de concentración conjunto, el que realmente sufre la ciudad.

Las tres fuentes usadas están definidas al inicio del script:

```python
fuentes = [
    (0, 15, 150, 200),   # Chuquicamata (principal)
    (-2, 7, 80, 150),    # Ministro Hales
    (1, 23, 100, 180),   # Radomiro Tomic
]
```

Cada tupla es `(xi, yi, A, k)`. Estos valores no son arbitrarios: se calibraron para que, evaluando el modelo en el centro de Calama `(0,0)`, la concentración de la fuente principal diera un valor cercano a 48,7 µg/m³ — es decir, cerca de la norma anual chilena de MP10 (50 µg/m³), lo que es coherente con que Calama haya sido declarada zona saturada. Cambiar esta lista (agregar, quitar o mover fuentes) es la forma de simular otros escenarios sin tocar el resto del programa; por eso está aislada al principio.

### 1.4 Gradiente (dirección de mayor crecimiento de la contaminación)

El gradiente de `C` en un punto es el vector de sus derivadas parciales, y apunta en la dirección en la que la concentración crece más rápido. Derivando la gaussiana con regla de la cadena se obtiene:

```
∂C/∂x = -2(x - xi)/k * C(x,y)
∂C/∂y = -2(y - yi)/k * C(x,y)
```

Notar que ambas derivadas son, en esencia, la propia concentración `C(x,y)` multiplicada por un factor que depende del desplazamiento respecto a la fuente. Como el factor `-2C/k` es siempre negativo, el gradiente **siempre apunta hacia la fuente emisora**: esa es la dirección en la que un observador vería aumentar la contaminación más rápido. Esto es clave para el análisis de riesgo (sección 5 del informe): indica hacia dónde alejarse para respirar mejor.


```python
def gradiente(x, y, A, k, xi, yi):
    C = concentracion(x, y, A, k, xi, yi)
    dCdx = -2 * (x - xi) / k * C
    dCdy = -2 * (y - yi) / k * C
    return dCdx, dCdy
```

La función reutiliza `concentracion(...)` en vez de recalcular la exponencial "a mano", para no duplicar la fórmula en dos lugares.

Se evalúa en el centro de la ciudad:

```python
dCdx, dCdy = gradiente(0, 0, A, k, xi, yi)
print(f"C(0,0)    = {concentracion(0, 0, A, k, xi, yi):.2f} ug/m3")
print(f"Gradiente = ({dCdx:.2f}, {dCdy:.2f})")
```

El resultado numérico es `C(0,0) ≈ 48.70` y `Gradiente ≈ (0.00, 7.31)`: como Chuquicamata está exactamente al norte del origen (misma coordenada `x = 0`), la componente horizontal del gradiente da cero, y la componente vertical (~7,31 µg/m³ por km) indica que moverse hacia el norte aumenta la concentración a esa tasa, mientras que moverse en el eje Este-Oeste no cambia nada localmente (esa dirección es tangente a la curva de nivel que pasa por el origen).

### 1.5 Integral doble: cantidad total y concentración promedio

Para saber cuánto contaminante hay "en total" sobre la región `R`, se integra la función de concentración en dos dimensiones:

```
M = ∬_R C(x,y) dA
```

 se aproxima con una **suma de Riemann**: se suma el valor de `C` en cada celda de la malla y se multiplica por el área de cada celda (`dx * dy`).

```python
dx = x[1] - x[0]
dy = y[1] - y[0]
M = np.sum(C_una) * dx * dy
area_R = (x[-1] - x[0]) * (y[-1] - y[0])
C_prom = M / area_R
```

- `dx`, `dy`: ancho de cada celda de la malla en cada eje. Como `linspace` genera puntos equiespaciados, basta restar los dos primeros valores para obtener el paso.
- `np.sum(C_una)`: suma todos los valores de concentración evaluados en los 400×400 puntos de la malla.
- `M = suma * dx * dy`: esto es exactamente la definición de una suma de Riemann — se aproxima el "volumen bajo la superficie" `C(x,y)` sumando el volumen de muchos prismas rectangulares muy delgados.
- `area_R`: el área del rectángulo `R` (`20 km × 30 km = 600 km²`), calculada directamente de los extremos de los vectores `x` e `y`.
- `C_prom = M / area_R`: la concentración promedio es la cantidad total dividida por el área, tal como se promedia cualquier función.

Cuantos más puntos tenga la malla (los 400 de `linspace`), más precisa es esta aproximación — es el mismo principio que "entre más rectángulos delgados, mejor se aproxima el área bajo una curva" que se ve en cálculo de una variable, extendido a dos dimensiones.

---

## 2. Recorrido del código, gráfico por gráfico

El script genera **5 figuras**, cada una respondiendo una pregunta distinta del análisis. Todas comparten los mismos ingredientes (la malla `X`, `Y` y los datos de `fuentes`), pero se usan de manera distinta.

### Gráfico 1 — Mapa de concentración (`mapa_concentracion.png`)

Muestra el campo `C_total` (las tres fuentes sumadas) como un mapa de calor:

```python
mapa = ax.contourf(X, Y, C_total, levels=20, cmap="YlOrRd")
curvas = ax.contour(X, Y, C_total, levels=[25, 50, 100], colors="black", linewidths=0.8)
ax.clabel(curvas, inline=True, fontsize=8)
```

- `contourf` (contour **f**illed) pinta el plano con 20 niveles de color, de amarillo claro a rojo oscuro, según la concentración.
- `contour` dibuja además tres curvas de nivel específicas en negro (25, 50 y 100 µg/m³), que son valores de referencia (50 es la norma anual chilena de MP10).
- `clabel` escribe el número directamente sobre cada curva para que se pueda leer sin tener que mirar la barra de colores.
- La estrella azul marca el centro de Calama y los cuadrados negros marcan cada fuente emisora, para dar contexto geográfico al mapa.

Este gráfico responde: *¿dónde están las zonas más contaminadas de la ciudad, considerando todas las fuentes a la vez?*

### Gráfico 2 — Superficie 3D (`superficie.png`)

```python
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot_surface(X, Y, C_total, cmap="YlOrRd")
```

Es el mismo campo `C_total`, pero graficado como una superficie tridimensional `z = C(x,y)` en vez de un mapa de calor visto "desde arriba". Los picos de la superficie corresponden visualmente a la ubicación de cada fuente (mientras más alto, más concentración). Este gráfico es más intuitivo para entender que el modelo es, literalmente, una suma de "montañas" gaussianas.

### Gráfico 3 — Curvas de nivel de una sola fuente (`curvas_nivel.png`)

```python
curvas2 = ax2.contour(X, Y, C_una, levels=[25, 50, 100],
                       colors=["green", "orange", "red"], linewidths=1.5)
```

A diferencia del Gráfico 1, aquí se usa `C_una` (solo la fuente principal, Chuquicamata), sin mapa de calor de fondo, para poder ver con claridad la forma exacta de las curvas de nivel: son **circunferencias concéntricas centradas en la fuente**, como se demuestra analíticamente en el informe (despejando `r² = k·ln(A/c)` de la ecuación del modelo). Este gráfico sirve para verificar visualmente esa deducción matemática.

### Gráfico 4 — Sensibilidad al parámetro `k` (`sensibilidad_k.png`)

Este gráfico (agregado como extensión del análisis base) responde: *¿qué tan sensible es la concentración en el centro de Calama a las condiciones atmosféricas (representadas por `k`)?*

```python
escenarios_k = {
    "Atenuada\n(k=120)": 120,
    "Base\n(k=200)": 200,
    "Expandida\n(k=350)": 350,
}
valores_k = [concentracion(0, 0, A, k_i, xi, yi) for k_i in escenarios_k.values()]
```

Se evalúa `C(0,0)` para la fuente principal manteniendo `A`, `xi`, `yi` fijos y variando solo `k`:

- `k = 120`: dispersión "atenuada" → simula alta estabilidad atmosférica (poco viento), la nube se concentra cerca de la fuente y en el centro de la ciudad baja la concentración.
- `k = 200`: escenario base del informe.
- `k = 350`: dispersión "expandida" → simula inversión térmica, la nube se extiende más lejos y en el centro de la ciudad sube la concentración.

El resultado se grafica como un gráfico de barras, con una línea horizontal punteada en 50 µg/m³ marcando la norma anual, para comparar de un vistazo cada escenario contra el límite legal:

```python
ax3.axhline(50, color="black", linestyle="--", linewidth=1, label="Norma anual MP10 (50 µg/m³)")
ax3.bar_label(barras_k, fmt="%.1f", padding=3, fontsize=11, fontweight="bold")
```

`bar_label` escribe el valor numérico encima de cada barra automáticamente, y `axhline` traza una línea horizontal de referencia — así no hace falta leer los ejes con precisión para saber si un escenario supera o no la norma.

### Gráfico 5 — Comparación una fuente vs. múltiples fuentes (`comparacion_fuentes.png`)

Responde la pregunta central del informe: *¿cuánto cambia la exposición de Calama si se considera solo Chuquicamata versus las tres faenas juntas?*

```python
C_una_centro = concentracion(0.0, 0.0, A, k, xi, yi)
C_total_centro = campo_total(0.0, 0.0, fuentes)
```

Aquí `campo_total` se evalúa con números sueltos (`0.0, 0.0`) en vez de la malla completa — la misma función sirve para ambos casos porque las operaciones de NumPy funcionan igual con escalares que con matrices. El resultado se grafica como barras, otra vez con la línea de la norma anual como referencia, mostrando que el efecto conjunto de las tres fuentes es notablemente mayor que el de la fuente principal sola.

Al final, el script imprime el aporte porcentual de cada fuente individual sobre el total en el centro de la ciudad:

```python
for (fxi, fyi, fA, fk) in fuentes:
    aporte = concentracion(0, 0, fA, fk, fxi, fyi)
    print(f"  ({fxi:>3}, {fyi:>3})  A={fA:<4} k={fk:<4} -> {aporte:6.2f} ug/m3  ({100*aporte/C_total_centro:.1f}%)")
```

Esto reveló un resultado interesante que el informe destaca: la faena **Ministro Hales**, aunque tiene menor emisión máxima (`A=80` vs. `A=150` de Chuquicamata), termina aportando más a la contaminación del centro porque está mucho más cerca (7,28 km vs. 15 km) — la distancia pesa más que la magnitud de la emisión en un modelo gaussiano, porque la distancia está al cuadrado dentro de la exponencial.


## 5. Cómo correr el código

### Requisitos

```bash
pip install numpy matplotlib
```

### Ejecución

```bash
python nombre_del_script.py
```
