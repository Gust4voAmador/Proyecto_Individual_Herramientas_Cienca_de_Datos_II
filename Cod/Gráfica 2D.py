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
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    return term1 + term2 + a + np.exp(1)


def funcion_cuadratica(x):
    # Coeficientes cuadráticos
    A = np.array([[2, 1], [1, 2]])  # Matriz de coeficientes cuadráticos (positiva definida)
    # Coeficientes lineales
    C = np.array([-6, -4])  # Vector de coeficientes lineales
    # Término constante
    D = 10  # Término constante

    # Calcular el valor de la función cuadrática
    resultado = 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(C, x) + D

    return resultado


# Graficar la función de aptitud
def plot_function(ax, funcion_aptitud, lim_inf, lim_sup):
    x = np.linspace(lim_inf[0], lim_sup[0], 100)
    y = np.linspace(lim_inf[1], lim_sup[1], 100)
    X, Y = np.meshgrid(x, y)
    Z = np.array([funcion_aptitud(np.array([x, y])) for x, y in zip(np.ravel(X), np.ravel(Y))])
    Z = Z.reshape(X.shape)
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.6)
    plt.colorbar(contour, ax=ax)



dimensiones = 2  # Usamos 2 dimensiones para poder graficar
lim_inf = np.array([-30] * dimensiones)
lim_sup = np.array([30] * dimensiones)
max_iter = 25  # Define el número máximo de iteraciones

mejor_posicion, mejor_valor, history = pso(funcion_cuadratica, dimensiones, lim_inf, lim_sup, max_iter=max_iter)

print("Mejor posición:", mejor_posicion)
print("Mejor valor:", mejor_valor)

# Graficar posiciones de partículas en iteraciones seleccionadas junto con la función
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

def plot_particles(ax, posiciones, iteracion):
    plot_function(ax, funcion_esfera, lim_inf, lim_sup)
    ax.scatter(posiciones[:, 0], posiciones[:, 1], c='red', marker='o', s=50)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    ax.set_title(f'Iteración {iteracion}')
    ax.grid(True)

plot_particles(axes[0], history["iter_0"], 'Inicial')
plot_particles(axes[1], history["iter_half"], f'{max_iter // 2}')
plot_particles(axes[2], history["iter_final"], 'Final')

plt.show()
