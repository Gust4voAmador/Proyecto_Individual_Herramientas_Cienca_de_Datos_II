# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:40:19 2024

@author: AMADOR
"""
import time
import numpy as np
import concurrent.futures

def pso(funcion_aptitud, dimensiones, lim_inf, lim_sup, n_particulas=30, w=0.5, c1=1.5, c2=1.5, max_iter=100):
    # Inicialización de posiciones y velocidades
    posiciones = np.random.uniform(lim_inf, lim_sup, (n_particulas, dimensiones))
    velocidades = np.random.uniform(-1, 1, (n_particulas, dimensiones))
    
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
        
        #if (gBest_valor - 0) < 2e-07:
        #    return gBest_posicion, gBest_valor, iteracion
                    
    return gBest_posicion, gBest_valor, max_iter

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

def ejecutar_pso_en_subregion(lim_inf, lim_sup, dimensiones):
    return pso(funcion_esfera, dimensiones, lim_inf, lim_sup)

def dividir_espacio_de_busqueda(lim_inf, lim_sup, num_subregiones):
    subregiones = []
    intervalos = np.linspace(lim_inf, lim_sup, num_subregiones + 1)
    for i in range(num_subregiones):
        subregiones.append((intervalos[i], intervalos[i + 1]))
    return subregiones


#Ejecutar el algoritmo de optimización

inicio = time.time()

if __name__ == '__main__':
    dimensiones = 10
    lim_inf_global = -30.0
    lim_sup_global = 30.0
    num_subregiones = 4  # Dividir el espacio de búsqueda en 4 subregiones

    # Crear los límites de cada subregión
    subregiones = dividir_espacio_de_busqueda(lim_inf_global, lim_sup_global, num_subregiones)

    # Ejecutar PSO en paralelo en cada subregión
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(ejecutar_pso_en_subregion, lim_inf, lim_sup, dimensiones) for lim_inf, lim_sup in subregiones]
        resultados = [future.result() for future in concurrent.futures.as_completed(futures)]

    # Encontrar la mejor solución entre todas las subregiones
    mejor_resultado = min(resultados, key=lambda x: x[1])
    
    print(f'Número de iteración: {mejor_resultado[2]}')
    print(f"Mejor posición: {mejor_resultado[0]}")
    print(f"Mejor valor: {mejor_resultado[1]}")
    fin = time.time()
    print(f'tiempo: {fin-inicio}')

