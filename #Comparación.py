import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# =========================
# constantes y parámetros
# =========================
c = 3e8
R = 12e3
I0 = 1e38

epsilon = 1e-6
alpha0 = np.deg2rad(45)
beta = 1e-12

omega0 = 187.5
omega_dot_obs0 = -2.37e-9

#calibramos K para reproducir omega punto inicial observado
K = -omega_dot_obs0 * (1 + 2 * epsilon * omega0**2) / (omega0**3 * np.sin(alpha0)**2)

print("K calibrado =", K)

# =========================
# ecuación diferencial
# =========================
def domega_dt(t, omega):
    alpha = alpha0 + beta * t
    return -K * omega**3 * (np.sin(alpha)**2) / (1 + 2 * epsilon * omega**2)

#formato para solve_ivp
def f_ivp(t, y):
    return [domega_dt(t, y[0])]

# =========================
# método RK4
# =========================
def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)

# =========================
# malla temporal uniforme
# =========================
t_max = 1e10
N_eval = 500
t = np.linspace(0, t_max, N_eval)

# =========================
# solución con RK4
# =========================
dt = t[1] - t[0]

omega_rk4 = np.zeros(N_eval)
omega_rk4[0] = omega0

for i in range(N_eval - 1):
    omega_rk4[i+1] = rk4_step(domega_dt, t[i], omega_rk4[i], dt)

# =========================
# solución con RK45
# =========================
sol = solve_ivp(
    fun=f_ivp,
    t_span=(0, t_max),
    y0=[omega0],
    method="RK45",
    t_eval=t,
    rtol=1e-9,
    atol=1e-12
)

omega_rk45 = sol.y[0]

# =========================
# error relativo
# =========================
error_rel = np.abs(omega_rk4 - omega_rk45) / (np.abs(omega_rk45) + 1e-300)

print("Error relativo máximo =", np.max(error_rel))
print("Error relativo promedio =", np.mean(error_rel))

# =========================
# cálculo de n para ambos métodos
# n = d ln|omega_punto| / d ln omega
# =========================
tiny = 1e-300

#RK4
domega_rk4 = domega_dt(t, omega_rk4)
lnOmega_rk4 = np.log(omega_rk4 + tiny)
lnAbsDomega_rk4 = np.log(np.abs(domega_rk4) + tiny)

d_lnOmega_rk4_dt = np.gradient(lnOmega_rk4, t)
d_lnAbsDomega_rk4_dt = np.gradient(lnAbsDomega_rk4, t)

n_rk4 = d_lnAbsDomega_rk4_dt / (d_lnOmega_rk4_dt + tiny)

#RK45
domega_rk45 = domega_dt(t, omega_rk45)
lnOmega_rk45 = np.log(omega_rk45 + tiny)
lnAbsDomega_rk45 = np.log(np.abs(domega_rk45) + tiny)

d_lnOmega_rk45_dt = np.gradient(lnOmega_rk45, t)
d_lnAbsDomega_rk45_dt = np.gradient(lnAbsDomega_rk45, t)

n_rk45 = d_lnAbsDomega_rk45_dt / (d_lnOmega_rk45_dt + tiny)

#recorte de bordes
k = 20
t_mid = t[k:-k]
n_rk4_mid = n_rk4[k:-k]
n_rk45_mid = n_rk45[k:-k]
error_mid = error_rel[k:-k]

print("n promedio RK4 =", np.mean(n_rk4_mid))
print("n promedio RK45 =", np.mean(n_rk45_mid))

# =========================
# gráficos
# =========================

#omega(t)
plt.figure()
plt.plot(t, omega_rk4, label="RK4")
plt.plot(t, omega_rk45, "--", label="RK45")
plt.xlabel("Tiempo (s)")
plt.ylabel("Omega (rad/s)")
plt.title("Comparación de Omega(t): RK4 vs RK45")
plt.legend()
plt.show()

#error relativo
plt.figure()
plt.plot(t_mid, error_mid)
plt.xlabel("Tiempo (s)")
plt.ylabel("Error relativo")
plt.title("Error relativo entre RK4 y RK45")
plt.show()

#índice de frenado
plt.figure()
plt.plot(t_mid, n_rk4_mid, label="n(t) RK4")
plt.plot(t_mid, n_rk45_mid, "--", label="n(t) RK45")
plt.axhline(2.51, color="gray", linestyle=":", label="n_obs ~ 2.51")
plt.xlabel("Tiempo (s)")
plt.ylabel("Índice de frenado n")
plt.title("Comparación del índice de frenado: RK4 vs RK45")
plt.legend()
plt.show()
