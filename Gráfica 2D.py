# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:21:45 2024

@author: AMADOR
"""
import numpy as np
import matplotlib.pyplot as plt

def initialize_particles(n_particulas, dimensiones, lim_inf, lim_sup):
    posiciones = np.random.uniform(lim_inf, lim_sup, (n_particulas, dimensiones))
    velocidades = np.random.uniform(-(lim_sup - lim_inf), lim_sup - lim_inf, (n_particulas, dimensiones))
    return posiciones, velocidades

def pso(funcion_aptitud, dimensiones, lim_inf, lim_sup, n_particulas=30, w=0.5, c1=1.5, c2=1.5, max_iter=100):
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

def funcion_esfera(x):
    return sum(x**2)

dimensiones = 2  # Usamos 2 dimensiones para poder graficar
lim_inf = np.array([-5.0] * dimensiones)
lim_sup = np.array([5.0] * dimensiones)
max_iter = 20  # Define el número máximo de iteraciones

mejor_posicion, mejor_valor, history = pso(funcion_esfera, dimensiones, lim_inf, lim_sup, max_iter=max_iter)

# Graficar posiciones de partículas en iteraciones seleccionadas
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_particles(ax, posiciones, iteracion):
    ax.scatter(posiciones[:, 0], posiciones[:, 1], c='blue', marker='o', s=50)
    ax.set_xlim(lim_inf[0], lim_sup[0])
    ax.set_ylim(lim_inf[1], lim_sup[1])
    ax.set_title(f'Iteración {iteracion}')
    ax.grid(True)

plot_particles(axes[0], history["iter_0"], 'Inicial')
plot_particles(axes[1], history["iter_half"], f'{max_iter // 2}')
plot_particles(axes[2], history["iter_final"], 'Final')

plt.show()
