# -*- coding: utf-8 -*-
"""
Created on Thu May 30 23:58:58 2024

@author: AMADOR
"""

import numpy as np
#Note que que los límites deben ser un array de la misma dimension qeu se indica
def initialize_particles(n_particulas, dimensiones, lim_inf, lim_sup):
    """
    Inicializa las posiciones y velocidades de las partículas.

    Args:
    n_particulas (int): Número de partículas.
    dimensiones (int): Dimensiones del espacio de búsqueda.
    lim_inf (array): Límite inferior del espacio de búsqueda.
    lim_sup (array): Límite superior del espacio de búsqueda.

    Returns:
    posiciones (array): Posiciones iniciales de las partículas.
    velocidades (array): Velocidades iniciales de las partículas.
    """
    # Inicializar posiciones de partículas dentro de los límites especificados crando una matriz
    posiciones = np.random.uniform(lim_inf, lim_sup, (n_particulas, dimensiones))
    # Inicializar velocidades de partículas dentro de un rango creando una matriz
    velocidades = np.random.uniform(-(lim_sup - lim_inf), lim_sup - lim_inf, (n_particulas, dimensiones))
    return posiciones, velocidades

#posiciones, velocidades = initialize_particles(5, 1,-5, 5)
#print(posiciones)
#print(velocidades)



def pso(funcion_aptitud, dimensiones, lim_inf, lim_sup, n_particulas=30, w=0.5, c1=1.5, c2=1.5, max_iter=100):
    """
    Implementa el algoritmo PSO estándar.

    Args:
    - funcion_aptitud: Función objetivo que se quiere minimizar.
    - dimensiones: Dimensiones del espacio de búsqueda.
    - lim_inf: Límite inferior del espacio de búsqueda.
    - lim_sup: Límite superior del espacio de búsqueda.
    - n_particulas: Número de partículas.
    - w: Factor de inercia.
    - c1: Coeficiente cognitivo (atracción hacia la mejor posición personal).
    - c2: Coeficiente social (atracción hacia la mejor posición global).
    - max_iter: Número máximo de iteraciones.

    Returns:
    - mejor_posicion_global: Mejor posición global encontrada.
    - mejor_valor_global: Valor de la función objetivo en la mejor posición global encontrada.
    """
    # Inicialización de posiciones y velocidades
    posiciones, velocidades = initialize_particles(n_particulas, dimensiones, lim_inf, lim_sup)
    # Inicialización de las mejores posiciones personales y sus valores de aptitud
    pBest_posiciones = np.copy(posiciones)
    pBest_valores = np.array([funcion_aptitud(p) for p in posiciones])
    # Encontrar la mejor posición global inicial y su valor de aptitud
    gBest_idx = np.argmin(pBest_valores)
    gBest_posicion = pBest_posiciones[gBest_idx]
    gBest_valor = pBest_valores[gBest_idx]
    
    # Iteración principal del PSO
    for iteracion in range(max_iter):
        for i in range(n_particulas):
            # Generar números aleatorios para las componentes cognitiva y social
            r1 = np.random.rand(dimensiones)
            r2 = np.random.rand(dimensiones)
            # Actualizar la velocidad de la partícula según la fórmula del PSO estándar
            velocidades[i] = (w * velocidades[i] + 
                              c1 * r1 * (pBest_posiciones[i] - posiciones[i]) + 
                              c2 * r2 * (gBest_posicion - posiciones[i]))
            # Actualizar la posición de la partícula
            posiciones[i] += velocidades[i]
            # Limitar las posiciones a los límites especificados
            posiciones[i] = np.clip(posiciones[i], lim_inf, lim_sup)
        
        # Evaluar la función objetivo y actualizar las mejores posiciones personales y globales
        for i in range(n_particulas):
            aptitud = funcion_aptitud(posiciones[i])
            # Actualizar la mejor posición personal (pBest) si la nueva posición es mejor
            if aptitud < pBest_valores[i]:
                pBest_posiciones[i] = posiciones[i]
                pBest_valores[i] = aptitud
                # Actualizar la mejor posición global (gBest) si la nueva posición es mejor
                if aptitud < gBest_valor:
                    gBest_posicion = posiciones[i]
                    gBest_valor = aptitud
    
    return gBest_posicion, gBest_valor

def funcion_esfera(x):
    return sum(x**2)

dimensiones = 3  # Cambia este valor para cualquier número de dimensiones
lim_inf = np.array([-5.0] * dimensiones)
lim_sup = np.array([5.0] * dimensiones)

mejor_posicion, mejor_valor = pso(funcion_esfera, dimensiones, lim_inf, lim_sup)
print(f"Mejor posición: {mejor_posicion}")
print(f"Mejor valor: {mejor_valor}")





