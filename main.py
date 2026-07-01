import numpy as np
import matplotlib.pyplot as plt

fuentes = [
    (0, 15, 150, 200),
    (-2, 7, 80, 150),
    (1, 23, 100, 180),
]

x = np.linspace(-10, 10, 400)
y = np.linspace(-5, 25, 400)
X, Y = np.meshgrid(x, y)


def concentracion(x, y, A, k, xi, yi):
    return A * np.exp(-((x - xi) ** 2 + (y - yi) ** 2) / k)


def campo_total(X, Y, fuentes):
    C = np.zeros_like(X)
    for (xi, yi, A, k) in fuentes:
        C += concentracion(X, Y, A, k, xi, yi)
    return C


def gradiente(x, y, A, k, xi, yi):
    C = concentracion(x, y, A, k, xi, yi)
    dCdx = -2 * (x - xi) / k * C
    dCdy = -2 * (y - yi) / k * C
    return dCdx, dCdy


xi, yi, A, k = fuentes[0][0], fuentes[0][1], fuentes[0][2], fuentes[0][3]
C_una = concentracion(X, Y, A, k, xi, yi)
C_total = campo_total(X, Y, fuentes)

dCdx, dCdy = gradiente(0, 0, A, k, xi, yi)
print(f"C(0,0)    = {concentracion(0, 0, A, k, xi, yi):.2f} ug/m3")
print(f"Gradiente = ({dCdx:.2f}, {dCdy:.2f})")

dx = x[1] - x[0]
dy = y[1] - y[0]
M = np.sum(C_una) * dx * dy
area_R = (x[-1] - x[0]) * (y[-1] - y[0])
C_prom = M / area_R
print(f"Cantidad total M    = {M:.2e} ug/m3 * km^2")
print(f"Concentracion media = {C_prom:.2f} ug/m3")

# ------------------------------------------------------------------
# GRÁFICO 1: mapa de concentración (3 fuentes)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 8))
mapa = ax.contourf(X, Y, C_total, levels=20, cmap="YlOrRd")
curvas = ax.contour(X, Y, C_total, levels=[25, 50, 100],
                     colors="black", linewidths=0.8)
ax.clabel(curvas, inline=True, fontsize=8)
fig.colorbar(mapa, ax=ax, label="Concentracion (ug/m3)")
ax.plot(0, 0, "b*", markersize=14, label="Calama (centro)")
for (fxi, fyi, fA, fk) in fuentes:
    ax.plot(fxi, fyi, "ks", markersize=7)
ax.set_xlabel("x (km, Este)")
ax.set_ylabel("y (km, Norte)")
ax.set_title("Dispersion de MP10 en Calama")
ax.legend()
plt.savefig("mapa_concentracion.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# GRÁFICO 2: superficie 3D
# ------------------------------------------------------------------
fig = plt.figure(figsize=(9, 7))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot_surface(X, Y, C_total, cmap="YlOrRd")
ax3d.set_xlabel("x (km)", labelpad=10)
ax3d.set_ylabel("y (km)", labelpad=10)
ax3d.set_zlabel("C (ug/m3)", labelpad=10)
ax3d.set_title("Superficie de concentracion")
fig.subplots_adjust(left=0.05, right=0.82, top=0.92, bottom=0.08)
plt.savefig("superficie.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# GRÁFICO 3: curvas de nivel (una fuente)
# ------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(7, 8))
curvas2 = ax2.contour(X, Y, C_una, levels=[25, 50, 100],
                       colors=["green", "orange", "red"], linewidths=1.5)
ax2.clabel(curvas2, inline=True, fontsize=8)
ax2.plot(0, 0, "b*", markersize=14, label="Calama (centro)")
for (fxi, fyi, fA, fk) in fuentes:
    ax2.plot(fxi, fyi, "ks", markersize=7)
ax2.set_xlabel("x (km, Este)")
ax2.set_ylabel("y (km, Norte)")
ax2.set_title("Curvas de nivel - MP10 en Calama")
ax2.legend()
plt.savefig("curvas_nivel.png", dpi=150, bbox_inches="tight")
plt.show()

# ------------------------------------------------------------------
# GRÁFICO 4 (NUEVO): sensibilidad del parámetro k en el centro de Calama
# ------------------------------------------------------------------
# Se evalúa C(0,0) para la fuente principal (Chuquicamata) variando k,
# mientras A, xi, yi se mantienen fijos.
escenarios_k = {
    "Atenuada\n(k=120)": 120,
    "Base\n(k=200)": 200,
    "Expandida\n(k=350)": 350,
}
valores_k = [concentracion(0, 0, A, k_i, xi, yi) for k_i in escenarios_k.values()]

fig3, ax3 = plt.subplots(figsize=(7, 5))
colores_k = ["#5FB878", "#F4B942", "#E5484D"]
barras_k = ax3.bar(escenarios_k.keys(), valores_k, color=colores_k, width=0.55)
ax3.axhline(50, color="black", linestyle="--", linewidth=1, label="Norma anual MP10 (50 µg/m³)")
ax3.bar_label(barras_k, fmt="%.1f", padding=3, fontsize=11, fontweight="bold")
ax3.set_ylabel("C(0,0)  [µg/m³]")
ax3.set_title("Sensibilidad de la concentración al parámetro de dispersión k")
ax3.set_ylim(0, max(valores_k) * 1.25)
ax3.legend()
plt.tight_layout()
plt.savefig("sensibilidad_k.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nSensibilidad de k en el centro de Calama:")
for (nombre, k_i), valor in zip(escenarios_k.items(), valores_k):
    print(f"  {nombre.replace(chr(10), ' ')}: {valor:.2f} ug/m3")

# ------------------------------------------------------------------
# GRÁFICO 5 (NUEVO): comparación 1 fuente vs múltiples fuentes
# ------------------------------------------------------------------
C_una_centro = concentracion(0.0, 0.0, A, k, xi, yi)          # solo Chuquicamata
C_total_centro = campo_total(0.0, 0.0, fuentes)                 # 3 fuentes superpuestas

escenarios_fuentes = ["1 fuente\n(Chuquicamata)", "3 fuentes\n(total)"]
valores_fuentes = [C_una_centro, C_total_centro]

fig4, ax4 = plt.subplots(figsize=(7, 5))
colores_f = ["#E08A4D", "#C1622D"]
barras_f = ax4.bar(escenarios_fuentes, valores_fuentes, color=colores_f, width=0.5)
ax4.axhline(50, color="black", linestyle="--", linewidth=1, label="Norma anual MP10 (50 µg/m³)")
ax4.bar_label(barras_f, fmt="%.1f", padding=3, fontsize=11, fontweight="bold")
ax4.set_ylabel("C(0,0)  [µg/m³]")
ax4.set_title("Comparación: una fuente vs. múltiples fuentes")
ax4.set_ylim(0, max(valores_fuentes) * 1.25)
ax4.legend()
plt.tight_layout()
plt.savefig("comparacion_fuentes.png", dpi=150, bbox_inches="tight")
plt.show()

print("\nComparacion 1 vs multiples fuentes:")
print(f"  1 fuente (Chuquicamata) : {C_una_centro:.2f} ug/m3")
print(f"  3 fuentes (total)       : {C_total_centro:.2f} ug/m3")

# Aporte porcentual de cada fuente en el centro (0,0)
print("\nAporte de cada fuente en el centro de Calama:")
for (fxi, fyi, fA, fk) in fuentes:
    aporte = concentracion(0, 0, fA, fk, fxi, fyi)
    print(f"  ({fxi:>3}, {fyi:>3})  A={fA:<4} k={fk:<4} -> {aporte:6.2f} ug/m3  ({100*aporte/C_total_centro:.1f}%)")