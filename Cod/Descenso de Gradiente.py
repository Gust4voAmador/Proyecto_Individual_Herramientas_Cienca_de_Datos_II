# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:48:16 2024

@author: AMADOR
"""

import time
from scipy.optimize import minimize
import numpy as np

inicio = time.time()

# Definir la función objetivo
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

# Generar el punto de inicio en el intervalo [-50, 50] para 10 dimensiones
dimensiones = 10
x0 = np.random.uniform(-5, 5, dimensiones)

# Usar scipy.optimize.minimize con la función esfera y método BFGS
resultado = minimize(funcion_esfera, x0, method='BFGS')

# Obtener el valor óptimo de la función en el punto encontrado
valor_optimo = funcion_esfera(resultado.x)

print("Resultado SciPy Descenso de Gradiente en 10 dimensiones:", resultado.x)
print("Valor de la función evaluada en el óptimo encontrado:", valor_optimo)

fin = time.time()

print(f'Tiempo de ejecución: {fin - inicio}')

