# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:00:46 2024

@author: AMADOR
"""

from scipy.optimize import dual_annealing
import numpy as np

# Ejemplo de uso en 5 dimensiones
func = lambda x: np.sum(x**2)  # Función objetivo
lim_inf = [-100] * 5
lim_sup = [100] * 5



def funcion_cuadratica(x):
    # Coeficientes cuadráticos
    A = np.array([[2, 5], [5, 3]])  # Matriz de coeficientes cuadráticos
    # Coeficientes lineales
    C = np.array([-6, -4])  # Vector de coeficientes lineales
    # Término constante
    D = 10  # Término constante

    # Calcular el valor de la función cuadrática
    resultado = 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(C, x) + D

    return resultado




resultado = dual_annealing(func, bounds=list(zip(lim_inf, lim_sup)))
print("Resultado SciPy Recocido Simulado en 5 dimensiones:", resultado.x)
