"""
modulo dónde irá rk4 y rk45 respectivamente
"""
import numpy as np
from typing import Callable

def rk4_method(f: Callable[[float, float], float], t0: float, y0: float, h: float, tf: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Método de Runge-Kutta de cuarto orden (RK4) para resolver ecuaciones diferenciales ordinarias (EDOs) de primer orden.

    Args:
        f (Callable[[float, float], float]): Función f(t, y) que define la EDO dy/dt = f(t, y).
                                             Debe aceptar dos flotantes y devolver la derivada.
        t0 (float): Valor inicial de la variable independiente
        y0 (float): Valor inicial de la variable dependiente (condición inicial).
        h (float): Tamaño del paso de integración
        tf (float): Valor final de la variable independiente

    Returns:
        tuple[np.ndarray, np.ndarray]: Una tupla (t, y) que contiene:
            - t (np.ndarray): Arreglo de valores discretos de la variable independiente.
            - y (np.ndarray): Valores aproximados de la variable dependiente calculados en cada t.
    """

    # variables independientes
    t = np.arange(start=t0, stop=tf + h, step=h, dtype=float)
    y = np.zeros_like(t, dtype=float)

    # Condición inicial
    y[0] = y0

    for n in range(len(t) - 1):
        tn = t[n]
        yn = y[n]

        k1 = h * f(tn, yn)
        k2 = h * f(tn + h/2, yn + k1/2)
        k3 = h * f(tn + h/2, yn + k2/2)
        k4 = h * f(tn + h, yn + k3)

        y[n+1] = yn + (1/6) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y

def rk45_method(f: Callable[[float, float], float], t0: float, y0: float, h: float, tf: float) -> tuple[np.ndarray, np.ndarray]:
    pass