import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#definimos constantes y parámetros del modelo
c = 3e8
R = 12e3
I0 = 1e38

epsilon = 1e-6
alpha0 = np.deg2rad(45)
beta = 1e-12

omega0 = 187.5
omega_dot_obs0 = -2.37e-9

#calibramos K para reproducir el valor inicial observado de omega punto
K = -omega_dot_obs0 * (1 + 2 * epsilon * omega0**2) / (omega0**3 * np.sin(alpha0)**2)

print("K calibrado =", K)

#ecuación diferencial para solve_ivp
def f_ivp(t, y):
    omega = y[0]
    alpha = alpha0 + beta * t
    domega = -K * omega**3 * (np.sin(alpha)**2) / (1 + 2 * epsilon * omega**2)
    return [domega]

#función omega punto para calcular n después
def domega_dt(t, omega):
    alpha = alpha0 + beta * t
    return -K * omega**3 * (np.sin(alpha)**2) / (1 + 2 * epsilon * omega**2)

#intervalo temporal
t_max = 1e10

#creamos una malla uniforme de tiempos para evaluar la solución
N_eval = 500
t_eval = np.linspace(0, t_max, N_eval)

#resolvemos con RK45
sol = solve_ivp(
    fun=f_ivp,
    t_span=(0, t_max),
    y0=[omega0],
    method="RK45",
    t_eval=t_eval,
    rtol=1e-9,
    atol=1e-12
)

#extraemos la solución
t_rk45 = sol.t
omega_rk45 = sol.y[0]

print("Omega(0) =", omega_rk45[0], "rad/s")
print("Omega(final) =", omega_rk45[-1], "rad/s")

# =========================
# cálculo del índice de frenado
# n = d ln|omega_punto| / d ln omega
# =========================

domega_rk45 = domega_dt(t_rk45, omega_rk45)

tiny = 1e-300

lnOmega = np.log(omega_rk45 + tiny)
lnAbsDomega = np.log(np.abs(domega_rk45) + tiny)

d_lnOmega_dt = np.gradient(lnOmega, t_rk45)
d_lnAbsDomega_dt = np.gradient(lnAbsDomega, t_rk45)

n_rk45 = d_lnAbsDomega_dt / (d_lnOmega_dt + tiny)

#recortamos más puntos en los bordes para evitar efectos numéricos
k = 20
t_mid = t_rk45[k:-k]
n_mid = n_rk45[k:-k]

print("n inicial ~", n_mid[0])
print("n promedio ~", np.mean(n_mid))

#gráfico de omega
plt.figure()
plt.plot(t_rk45, omega_rk45)
plt.xlabel("Tiempo (s)")
plt.ylabel("Omega (rad/s)")
plt.title("Frenado rotacional (RK45)")
plt.show()

#gráfico de n
plt.figure()
plt.plot(t_mid, n_mid)
plt.xlabel("Tiempo (s)")
plt.ylabel("Índice de frenado n")
plt.title("Índice de frenado (RK45)")
plt.show()

#=========================
#desaceleración
#=========================

omega_dot_rk45 = domega_dt(t_rk45, omega_rk45)
print("Omega_dot(0) =", omega_dot_rk45[0], "rad/s^2")
print("Omega_dot(final) =", omega_dot_rk45[-1], "rad/s^2")
plt.figure()
plt.plot(t_rk45, omega_dot_rk45)
plt.xlabel("Tiempo (s)")
plt.ylabel("Omega punto (rad/s^2)")
plt.title("Desaceleración con RK45")
plt.show()
