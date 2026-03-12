#Comparación

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# parámetros del modelo

epsilon = 1e-6
alpha0 = np.deg2rad(45)
beta = 1e-12

# condiciones iniciales (Cangrejo)
omega0 = 187.5
omega_dot_obs0 = -2.37e-9  # enunciado

# calibración del prefactor K para que domega_dt(0, omega0) = omega_dot_obs0

K = -omega_dot_obs0 * (1 + 2*epsilon*omega0**2) / (omega0**3 * np.sin(alpha0)**2)

def domega_dt(t, omega):
    alpha = alpha0 + beta * t
    return -K * omega**3 * (np.sin(alpha)**2) / (1 + 2 * epsilon * omega**2)

def f_ivp(t, y):
    return [domega_dt(t, y[0])]

# RK4 paso fijo

def rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + h/2, y + h*k1/2)
    k3 = f(t + h/2, y + h*k2/2)
    k4 = f(t + h, y + h*k3)
    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

# tiempo de simulación

t_max = 1e10      # s (~317 años) para que se vea cambio sin ser enorme
dt = 1e6          # s (RK4)
N = int(t_max/dt) + 1
t_grid = np.linspace(0.0, t_max, N)

print("Omega_dot(0) modelo =", domega_dt(0.0, omega0), "rad/s^2")

# 1) integrar con RK4

omega_rk4 = np.zeros(N)
omega_rk4[0] = omega0
for i in range(N - 1):
    omega_rk4[i+1] = rk4_step(domega_dt, t_grid[i], omega_rk4[i], dt)

# 2) integrar con RK45 (solve_ivp)

sol = solve_ivp(
    fun=f_ivp,
    t_span=(0.0, t_max),
    y0=[omega0],
    method="RK45",
    t_eval=t_grid,   # misma malla -> comparación directa
    rtol=1e-9,
    atol=1e-12
)
if not sol.success:
    raise RuntimeError(sol.message)

omega_rk45 = sol.y[0]

# comparación

diff = omega_rk4 - omega_rk45
rel_err = np.abs(diff) / (np.abs(omega_rk45) + 1e-30)

print("\n=== Comparación RK4 vs RK45 ===")
print("max |Ω4-Ω45| =", np.max(np.abs(diff)))
print("max error relativo =", np.max(rel_err))

# gráficos

plt.figure()
plt.plot(t_grid, omega_rk4, label="RK4")
plt.plot(t_grid, omega_rk45, linestyle="--", label="RK45")
plt.xlabel("Tiempo (s)")
plt.ylabel("Omega (rad/s)")
plt.title("Omega(t): RK4 vs RK45 (K calibrado)")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_grid, rel_err)
plt.xlabel("Tiempo (s)")
plt.ylabel("Error relativo |Ω4-Ω45|/|Ω45|")
plt.title("Error relativo RK4 vs RK45")
plt.show()