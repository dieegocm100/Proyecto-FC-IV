#Método RK4 (2)

import numpy as np
import matplotlib.pyplot as plt

#definimos las constantes dadas en el enucniado, correspondientes al púlsar del Cangrejo
#se asumen las constantes alpha_0 = 45° y beta = 1e-12 rad/s que aparecen en P_dip. Alpha es el ángulo 
#entre el eje magnético y el eje de rotación, y beta es la tasa de cambio de ese ángulo. 
#Se asume que el ángulo aumenta linealmente con el tiempo, lo que es una simplificación común en modelos de frenado rotacional.
#Con esto, simplificamos el modelo para que dependa solo de omega y no del cambio temporal de alpha y beta, lo que facilita la integración numérica.

c = 3e8
R = 12e3
I0 = 1e38

#parámetros 

epsilon = 1e-6
alpha0 = np.deg2rad(45)
beta = 1e-12

omega0 = 187.5
omega_dot_obs0 = -2.37e-9  # enunciado

K = -omega_dot_obs0 * (1 + 2*epsilon*omega0**2) / (omega0**3 * np.sin(alpha0)**2)
print("K calibrado =", K)

#creamos la función omega punto

def domega_dt(t, omega):
    alpha = alpha0 + beta * t
    return -K * omega**3 * (np.sin(alpha)**2) / (1 + 2 * epsilon * omega**2)

#método RK4

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

#definimos los valores iniciales y el rango de tiempo para la integración

omega0 = 187.5      # rad/s
t_max = 1e10      # s (~317 años)
dt = 1e7            # s. 4 meses de paso.
N = int(t_max/dt) + 1

t = np.linspace(0, t_max, N)
omega = np.zeros(N)
omega[0] = omega0

for i in range(N - 1):
    omega[i+1] = rk4_step(domega_dt, t[i], omega[i], dt)

#graficamos

print("Omega(0)      =", omega[0], "rad/s")
print("Omega(final)  =", omega[-1], "rad/s")
print("Omega_dot(0)  =", domega_dt(0.0, omega0), "rad/s^2")

plt.plot(t, omega)
plt.xlabel("Tiempo (s)")
plt.ylabel("Omega (rad/s)")
plt.title("Frenado rotacional (RK4)")
plt.show()

# =========================
# calcular índice de frenado n
# =========================

# Omega punto usando el modelo
domega = domega_dt(t, omega)

tiny = 1e-300

lnOmega = np.log(omega + tiny)
lnAbsDomega = np.log(np.abs(domega) + tiny)

d_lnOmega_dt = np.gradient(lnOmega, t)
d_lnAbsDomega_dt = np.gradient(lnAbsDomega, t)

n = d_lnAbsDomega_dt / (d_lnOmega_dt + tiny)

# quitamos bordes donde np.gradient es inestable
k = 5
t_mid = t[k:-k]
n_mid = n[k:-k]

print("n inicial ~", n_mid[0])
print("n promedio ~", np.mean(n_mid))

plt.figure()
plt.plot(t_mid, n_mid)
plt.xlabel("Tiempo (s)")
plt.ylabel("Índice de frenado n")
plt.title("Evolución del índice de frenado (RK4)")
plt.show()

# =========================
# desaceleración con RK4
# =========================
omega_dot = domega_dt(t, omega)

print("Omega_dot(0) =", omega_dot[0], "rad/s^2")
print("Omega_dot(final) =", omega_dot[-1], "rad/s^2")

plt.figure()
plt.plot(t, omega_dot)
plt.xlabel("Tiempo (s)")
plt.ylabel("Omega punto (rad/s^2)")
plt.title("Desaceleración con RK4")
plt.show()
