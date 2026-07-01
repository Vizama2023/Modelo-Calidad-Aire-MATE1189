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


# ---- Fuente 1 (para C_una, gradiente e integral) ----
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

# ---- Gráfico 1: mapa de contorno relleno (campo total) ----
fig, ax = plt.subplots(figsize=(7, 8))
mapa = ax.contourf(X, Y, C_total, levels=20, cmap="YlOrRd")
curvas = ax.contour(X, Y, C_total, levels=[25, 50, 100],
                     colors="black", linewidths=0.8)
ax.clabel(curvas, inline=True, fontsize=8)
fig.colorbar(mapa, ax=ax, label="Concentracion (ug/m3)")
ax.plot(0, 0, "b*", markersize=14, label="Calama (centro)")
for (xi, yi, A, k) in fuentes:
    ax.plot(xi, yi, "ks", markersize=7)
ax.set_xlabel("x (km, Este)")
ax.set_ylabel("y (km, Norte)")
ax.set_title("Dispersion de MP10 en Calama")
ax.legend()
plt.savefig("mapa_concentracion.png", dpi=150, bbox_inches="tight")
plt.show()

# ---- Gráfico 2: superficie 3D ----
# FIX: en gráficos 3D, bbox_inches="tight" no calcula bien el espacio de la
# etiqueta rotada del eje Z y la corta. Se reemplaza por un margen manual
# con subplots_adjust(), sin bbox_inches="tight".
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

# ---- Gráfico 3: curvas de nivel (solo fuente 1) ----
fig2, ax2 = plt.subplots(figsize=(7, 8))
curvas2 = ax2.contour(X, Y, C_una, levels=[25, 50, 100],
                       colors=["green", "orange", "red"], linewidths=1.5)
ax2.clabel(curvas2, inline=True, fontsize=8)
ax2.plot(0, 0, "b*", markersize=14, label="Calama (centro)")
for (xi, yi, A, k) in fuentes:
    ax2.plot(xi, yi, "ks", markersize=7)
ax2.set_xlabel("x (km, Este)")
ax2.set_ylabel("y (km, Norte)")
ax2.set_title("Curvas de nivel - MP10 en Calama")
ax2.legend()
plt.savefig("curvas_nivel.png", dpi=150, bbox_inches="tight")
plt.show()