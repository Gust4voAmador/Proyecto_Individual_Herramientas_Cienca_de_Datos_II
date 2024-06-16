# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 16:05:44 2024

@author: AMADOR
"""
from ejemplo_paralelizacion import Paralelizacion

# Ejemplo de uso
if __name__ == '__main__':
    paralelizacion = Paralelizacion(dimensiones=10, lim_inf_global=-300.0, lim_sup_global=300.0, num_subregiones=6)
    paralelizacion.ejecutar_optimizacion()
