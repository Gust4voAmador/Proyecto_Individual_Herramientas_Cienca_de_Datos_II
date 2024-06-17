# -*- coding: utf-8 -*-
"""
Created on Fri May 31 12:04:21 2024

@author: AMADOR
"""

import numpy as np

# Función Sphere
def sphere(x):
    return sum(x**2)

# Función Rosenbrock
def rosenbrock(x):
    return sum(100 * (x[1:] - x[:-1]**2)**2 + (x[:-1] - 1)**2)

# Función Ackley
def ackley(x):
    a = 20
    b = 0.2
    c = 2 * np.pi
    d = len(x)
    sum_sq_term = -a * np.exp(-b * np.sqrt(sum(x**2) / d))
    cos_term = -np.exp(sum(np.cos(c * x) / d))
    return a + np.exp(1) + sum_sq_term + cos_term

# Función Griewank
def griewank(x):
    sum_sq_term = sum(x**2) / 4000
    prod_cos_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
    return sum_sq_term - prod_cos_term + 1

# Función Rastrigin
def rastrigin(x):
    A = 10
    return A * len(x) + sum(x**2 - A * np.cos(2 * np.pi * x))

# Función PSO (mantiene la implementación previa)
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
    
    return gBest_posicion, gBest_valor

# Parámetros comunes
dimensiones = 3  # Puedes cambiar este valor según sea necesario
limites = {
    "sphere": (-500, 500),
    "rosenbrock": (-2.048, 2.048),
    "ackley": (-32, 32),
    "griewank": (-600, 600),
    "rastrigin": (-5.12, 5.12)
}

# Probar las funciones de prueba
funciones = [sphere, rosenbrock, ackley, griewank, rastrigin]
nombres = ["Sphere", "Rosenbrock", "Ackley", "Griewank", "Rastrigin"]

for func, nombre in zip(funciones, nombres):
    lim_inf = np.array([limites[nombre.lower()][0]] * dimensiones)
    lim_sup = np.array([limites[nombre.lower()][1]] * dimensiones)
    mejor_posicion, mejor_valor = pso(func, dimensiones, lim_inf, lim_sup)
    print(f"Función: {nombre}")
    print(f"Mejor posición: {mejor_posicion}")
    print(f"Mejor valor: {mejor_valor}")
    print("-" * 30)
