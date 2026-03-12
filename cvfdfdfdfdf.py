import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# parametros del crab
Omega0 = 190        # rad/s
n = 2.5
K = 1e-15

dt = 1000
steps = 2000

Omega = Omega0

omega_list = []

# integrar ecuacion
for i in range(steps):

    Omega_dot = -K * Omega**n
    Omega = Omega + Omega_dot*dt

    omega_list.append(Omega)

omega_list = np.array(omega_list)

fig, ax = plt.subplots()

line, = ax.plot([],[], lw=3)

def update(frame):

    angle = omega_list[frame]*frame*0.001

    x = np.cos(angle)
    y = np.sin(angle)

    line.set_data([0,x],[0,y])

    return line,

ani = FuncAnimation(fig, update, frames=1000)

ani.save("pulsar_spindown.gif", fps=30)