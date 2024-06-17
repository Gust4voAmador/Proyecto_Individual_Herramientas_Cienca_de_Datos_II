# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 02:04:37 2024

@author: AMADOR
"""

import time
import numpy as np
from scipy.optimize import minimize

# Definir la función de Ackley
def ackley_function(x):
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



# Implementación de PSO
def initialize_particles(n_particles, dimensions, lim_inf, lim_sup):
    posiciones = np.random.uniform(lim_inf, lim_sup, (n_particles, dimensions))
    velocidades = np.random.uniform(-(lim_sup - lim_inf), lim_sup - lim_inf, (n_particles, dimensions))
    return posiciones, velocidades

def pso(func, dimensiones, lim_inf, lim_sup, n_particles=30, w=0.5, c1=1.5, c2=1.5, max_iter=100):
    posiciones, velocidades = initialize_particles(n_particles, dimensiones, lim_inf, lim_sup)
    pBest_posiciones = np.copy(posiciones)
    pBest_valores = np.array([func(p) for p in posiciones])
    
    gBest_idx = np.argmin(pBest_valores)
    gBest_posicion = pBest_posiciones[gBest_idx]
    gBest_valor = pBest_valores[gBest_idx]
    
    for iteracion in range(max_iter):
        for i in range(n_particles):
            r1 = np.random.rand(dimensiones)
            r2 = np.random.rand(dimensiones)
            velocidades[i] = (w * velocidades[i] + 
                              c1 * r1 * (pBest_posiciones[i] - posiciones[i]) + 
                              c2 * r2 * (gBest_posicion - posiciones[i]))
            posiciones[i] += velocidades[i]
            posiciones[i] = np.clip(posiciones[i], lim_inf, lim_sup)
        
        for i in range(n_particles):
            aptitud = func(posiciones[i])
            if aptitud < pBest_valores[i]:
                pBest_posiciones[i] = posiciones[i]
                pBest_valores[i] = aptitud
                if aptitud < gBest_valor:
                    gBest_posicion = posiciones[i]
                    gBest_valor = aptitud
    
    return gBest_posicion, gBest_valor

# Comparar PSO y Descenso de Gradiente en la función de Ackley

dimensiones = 2
lim_inf = np.array([-500.0] * dimensiones)
lim_sup = np.array([500.0] * dimensiones)

# Usar scipy.optimize.minimize con la función de Ackley y método BFGS
x0 = np.random.uniform(-5, 5, dimensiones)
inicio_gradiente = time.time()
resultado_gradiente = minimize(funcion_cuadratica, x0, method='BFGS')
fin_gradiente = time.time()

print("Resultado SciPy Descenso de Gradiente:", resultado_gradiente.x)
print('')
print("Valor de la función evaluada en el óptimo encontrado:", ackley_function(resultado_gradiente.x))
print(f'Tiempo de ejecución (Descenso de Gradiente): {fin_gradiente - inicio_gradiente}')

# Usar PSO
inicio_pso = time.time()
mejor_posicion_pso, mejor_valor_pso = pso(funcion_cuadratica, dimensiones, lim_inf, lim_sup)
fin_pso = time.time()

print("Resultado PSO:", mejor_posicion_pso)
print('')
print("Valor de la función evaluada en el óptimo encontrado:", mejor_valor_pso)
print(f'Tiempo de ejecución (PSO): {fin_pso - inicio_pso}')
