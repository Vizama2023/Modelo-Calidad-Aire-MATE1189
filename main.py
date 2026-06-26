import numpy as np            # calculo numerico y arreglos
import matplotlib.pyplot as plt  # graficos 2D y 3D

fuentes = [
    (0, 15, 150, 200),   # Chuquicamata (fuente principal)
    (-2, 7, 80, 150),    # Ministro Hales
    (1, 23, 100, 180),   # Radomiro Tomic
]

x = np.linspace(-10, 10, 400)   # eje Este-Oeste
y = np.linspace(-5, 25, 400)    # eje Norte-Sur
X, Y = np.meshgrid(x, y)        # malla de coordenadas (X, Y)

def concentracion(x, y, A, k, xi, yi):
    # modelo: C = A * exp( -((x-xi)^2 + (y-yi)^2) / k )
    return A * np.exp(-((x - xi)**2 + (y - yi)**2) / k)

def campo_total(X, Y, fuentes):
    C = np.zeros_like(X)          # arreglo de ceros del tamano de la malla
    for (xi, yi, A, k) in fuentes:
        C += concentracion(X, Y, A, k, xi, yi)  # se suma cada fuente
    return C

def gradiente(x, y, A, k, xi, yi):
    C = concentracion(x, y, A, k, xi, yi)
    dCdx = -2 * (x - xi) / k * C   # derivada parcial respecto de x
    dCdy = -2 * (y - yi) / k * C   # derivada parcial respecto de y
    return dCdx, dCdy

xi, yi, A, k = fuentes[0][0], fuentes[0][1], fuentes[0][2], fuentes[0][3]
C_una = concentracion(X, Y, A, k, xi, yi)   # campo de la fuente principal
C_total = campo_total(X, Y, fuentes)        # campo de todas las fuentes

dCdx, dCdy = gradiente(0, 0, A, k, xi, yi)
print(f"C(0,0)    = {concentracion(0, 0, A, k, xi, yi):.2f} ug/m3")
print(f"Gradiente = ({dCdx:.2f}, {dCdy:.2f})")

dx = x[1] - x[0]                 # ancho de celda en x
dy = y[1] - y[0]                 # ancho de celda en y
M = np.sum(C_una) * dx * dy      # cantidad total (integral doble aproximada)
area_R = (x[-1] - x[0]) * (y[-1] - y[0])  # area de R = 600 km^2
C_prom = M / area_R              # concentracion promedio
print(f"Cantidad total M    = {M:.2e} ug/m3 * km^2")
print(f"Concentracion media = {C_prom:.2f} ug/m3")

fig, ax = plt.subplots(figsize=(7, 8))
mapa = ax.contourf(X, Y, C_total, levels=20, cmap="YlOrRd")  # mapa de calor
curvas = ax.contour(X, Y, C_total, levels=[25, 50, 100],
                    colors="black", linewidths=0.8)          # curvas de nivel
ax.clabel(curvas, inline=True, fontsize=8)                   # etiquetas
fig.colorbar(mapa, ax=ax, label="Concentracion (ug/m3)")
ax.plot(0, 0, "b*", markersize=14, label="Calama (centro)")  # ciudad
for (xi, yi, A, k) in fuentes:
    ax.plot(xi, yi, "ks", markersize=7)                      # fuentes
ax.set_xlabel("x (km, Este)")
ax.set_ylabel("y (km, Norte)")
ax.set_title("Dispersion de MP10 en Calama")
ax.legend()
plt.savefig("mapa_concentracion.png", dpi=150, bbox_inches="tight")
plt.show()

fig = plt.figure(figsize=(8, 6))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot_surface(X, Y, C_total, cmap="YlOrRd")  # superficie 3D
ax3d.set_xlabel("x (km)")
ax3d.set_ylabel("y (km)")
ax3d.set_zlabel("C (ug/m3)")
ax3d.set_title("Superficie de concentracion")
plt.savefig("superficie.png", dpi=150, bbox_inches="tight")
plt.show()

# Curvas de nivel por separado (Figura 2 del informe)
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