# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:18:45 2024

@author: AMADOR
"""

import numpy as np
import matplotlib.pyplot as plt

def initialize_particles(n_particulas, dimensiones, lim_inf, lim_sup):
    posiciones = np.random.uniform(lim_inf, lim_sup, (n_particulas, dimensiones))
    velocidades = np.random.uniform(-(lim_sup - lim_inf), lim_sup - lim_inf, (n_particulas, dimensiones))
    return posiciones, velocidades

def pso(funcion_aptitud, dimensiones, lim_inf, lim_sup, n_particulas=30, w=0.5, c1=1.5, c2=1.5, max_iter=6000):
    posiciones, velocidades = initialize_particles(n_particulas, dimensiones, lim_inf, lim_sup)
    pBest_posiciones = np.copy(posiciones)
    pBest_valores = np.array([funcion_aptitud(p) for p in posiciones])
    gBest_idx = np.argmin(pBest_valores)
    gBest_posicion = pBest_posiciones[gBest_idx]
    gBest_valor = pBest_valores[gBest_idx]
    
    history = {"iter_0": np.copy(posiciones)}
    
    for iteracion in range(max_iter):
        for i in range(n_particulas):
            r1 = np.random.rand(dimensiones)
            r2 = np.random.rand(dimensiones)
            velocidades[i] = (w * velocidades[i] + 
                              c1 * r1 * (pBest_posiciones[i] - posiciones[i]) + 
                              c2 * r2 * (gBest_posicion - posiciones[i]))
            posiciones[i] += velocidades[i]
            posiciones[i] = np.clip(posiciones[i], lim_inf, lim_sup)
        
        for i in range(n_particulas):
            aptitud = funcion_aptitud(posiciones[i])
            if aptitud < pBest_valores[i]:
                pBest_posiciones[i] = posiciones[i]
                pBest_valores[i] = aptitud
                if aptitud < gBest_valor:
                    gBest_posicion = posiciones[i]
                    gBest_valor = aptitud
        
        if iteracion == max_iter // 2:
            history["iter_half"] = np.copy(posiciones)
    
    history["iter_final"] = np.copy(posiciones)
    return gBest_posicion, gBest_valor, history

def funcion_esfera_1d(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = 1  # Para 1 dimensión
    sum1 = x**2
    sum2 = np.cos(c * x)
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)

dimensiones = 1  # Usamos 1 dimensión para graficar en 1D
lim_inf = np.array([-200] * dimensiones)
lim_sup = np.array([200] * dimensiones)
max_iter = 25  # Define el número máximo de iteraciones

mejor_posicion, mejor_valor, history = pso(funcion_esfera_1d, dimensiones, lim_inf, lim_sup, max_iter=max_iter)

print("Mejor posición:", mejor_posicion)
print("Mejor valor:", mejor_valor)

# Graficar la función de aptitud y las posiciones de las partículas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_particles_1d(ax, posiciones, iteracion):
    x = np.linspace(lim_inf[0], lim_sup[0], 400)
    y = np.array([funcion_esfera_1d(xi) for xi in x])
    ax.plot(x, y, 'b-', label='Función de aptitud')
    ax.scatter(posiciones, funcion_esfera_1d(posiciones), c='red', marker='o', s=50, label='Partículas')
    ax.set_xlim(-1, 20)
    ax.set_ylim(- 1, 20)
    ax.set_title(f'Iteración {iteracion}')
    ax.grid(True)
    ax.legend()

plot_particles_1d(axes[0], history["iter_0"], 'Inicial')
plot_particles_1d(axes[1], history["iter_half"], f'{max_iter // 2}')
plot_particles_1d(axes[2], history["iter_final"], 'Final')

plt.show()
