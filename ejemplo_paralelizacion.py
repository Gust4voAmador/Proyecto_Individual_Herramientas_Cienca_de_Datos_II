# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 21:53:28 2024

@author: AMADOR
"""
import time
import numpy as np
import concurrent.futures

class Paralelizacion:
    def __init__(self, dimensiones=2, lim_inf_global=-10.0, lim_sup_global=10.0, num_subregiones=6, funcion='ackley'):
        self.dimensiones = dimensiones
        self.lim_inf_global = lim_inf_global
        self.lim_sup_global = lim_sup_global
        self.num_subregiones = num_subregiones
        self.funcion = funcion
    
    def pso(self, funcion_aptitud, lim_inf, lim_sup, n_particulas=30, w=0.5, c1=1.5, c2=1.5, max_iter=100):
        # Inicialización de posiciones y velocidades
        posiciones = np.random.uniform(lim_inf, lim_sup, (n_particulas, self.dimensiones))
        velocidades = np.random.uniform(-1, 1, (n_particulas, self.dimensiones))
        
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
                r1 = np.random.rand(self.dimensiones)
                r2 = np.random.rand(self.dimensiones)
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
                    
        return gBest_posicion, gBest_valor, max_iter

    def funcion_ackley(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
        term2 = -np.exp(sum2 / d)
        return term1 + term2 + a + np.exp(1)

    def funcion_cuadratica(self, x):
        # Coeficientes cuadráticos
        A = np.array([[2, 1], [1, 2]])  # Matriz de coeficientes cuadráticos (positiva definida)
        # Coeficientes lineales
        C = np.array([-6, -4])  # Vector de coeficientes lineales
        # Término constante
        D = 10  # Término constante

        # Calcular el valor de la función cuadrática
        resultado = 0.5 * np.dot(x.T, np.dot(A, x)) + np.dot(C, x) + D

        return resultado

    def ejecutar_pso_en_subregion(self, lim_inf, lim_sup, funcion, n_particulas, max_iter):
        if funcion == 'ackley':
            return self.pso(self.funcion_ackley, lim_inf, lim_sup, n_particulas=n_particulas, max_iter=max_iter)
        elif funcion == 'cuadratica':
            if self.dimensiones != 2:
                raise ValueError("La cuadrática solo acepta 2 dimensiones.")
            return self.pso(self.funcion_cuadratica, lim_inf, lim_sup, n_particulas=n_particulas, max_iter=max_iter)
        else:
            raise ValueError("Función ingresada inválida.")
        
    def dividir_espacio_de_busqueda(self):
        subregiones = []
        intervalos = np.linspace(self.lim_inf_global, self.lim_sup_global, self.num_subregiones + 1)
        for i in range(self.num_subregiones):
            subregiones.append((intervalos[i], intervalos[i + 1]))
        return subregiones

    def ejecutar_optimizacion(self, n_particulas=30, max_iter=100):
        inicio = time.time()

        # Crear los límites de cada subregión
        subregiones = self.dividir_espacio_de_busqueda()

        # Ejecutar PSO en paralelo en cada subregión
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.ejecutar_pso_en_subregion, lim_inf, lim_sup, self.funcion, n_particulas, max_iter) for lim_inf, lim_sup in subregiones]
            resultados = [future.result() for future in concurrent.futures.as_completed(futures)]

        # Encontrar la mejor solución entre todas las subregiones
        mejor_resultado = min(resultados, key=lambda x: x[1])
        fin = time.time()
        
        mejor_valor = mejor_resultado[1]
        tiempo = fin - inicio
        return mejor_valor, tiempo
        #print(f'Número de iteración: {mejor_resultado[2]}')
        #print(f"Mejor posición: {mejor_resultado[0]}")
        #print(f"Mejor valor: {mejor_resultado[1]}")
        
        #print(f'Tiempo transcurrido: {fin - inicio} segundos')

# Ejemplo de uso
if __name__ == '__main__':
    paralelizacion = Paralelizacion(dimensiones=5, lim_inf_global=-30.0, lim_sup_global=30.0, num_subregiones=6, funcion='ackley')
    paralelizacion.ejecutar_optimizacion(n_particulas=50, max_iter=100)
    