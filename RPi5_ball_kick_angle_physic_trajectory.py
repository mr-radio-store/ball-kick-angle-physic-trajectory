import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# =====================================================
# Estimate angle with ball kicked under physic trajectory
# 1. Soccer Ball Parameters
# =====================================================
m = 0.43           # kg
r_ball = 0.11      # meters
A = np.pi * r_ball**2  # cross-sectional area

g = 9.81           # gravity [m/s²]
rho_air = 1.2      # air density [kg/m³]
Cd = 0.25          # drag coefficient
Cl = 0.15          # Magnus coefficient

dt = 0.01
max_time = 10.0

# =====================================================
# 2. Simulation Parameters
# =====================================================
num_simulations = 30  # number of kicks

# Kick variations
speed_mean = 25.0
speed_sigma = 2.0

angle_mean = 20.0
angle_sigma = 5.0

spin_mean = 30.0   # rad/s
spin_sigma = 5.0

# Wind variations (x, y, z)
wind_mean = [0.0, 0.0, 0.0]   # no wind
wind_sigma = [2.0, 1.0, 0.0]  # max ± variation in m/s

# =====================================================
# 3. Simulation Function
# =====================================================
def simulate_kick(v0, angle_deg, omega_spin):
    # Convert angle to radians
    angle_rad = np.radians(angle_deg)

    # Initial velocity components
    vx = v0 * np.cos(angle_rad)
    vy = 0.0
    vz = v0 * np.sin(angle_rad)
    v = np.array([vx, vy, vz])

    # Spin vector along y-axis (side curve)
    omega = np.array([0.0, omega_spin, 0.0])

    # Random wind
    wind = np.array([np.random.normal(wind_mean[0], wind_sigma[0]),
                     np.random.normal(wind_mean[1], wind_sigma[1]),
                     0.0])  # assume wind horizontal only

    r = np.array([0.0, 0.0, 0.0])  # starting position
    traj = [r.copy()]
    t = 0.0

    while r[2] >= 0 and t < max_time:
        speed_rel = v - wind  # velocity relative to air
        speed = np.linalg.norm(speed_rel)
        if speed == 0:
            break
        v_hat = speed_rel / speed
        omega_hat = omega / (np.linalg.norm(omega)+1e-8)

        # Forces
        Fg = np.array([0, 0, -m*g])
        Fd = -0.5 * rho_air * Cd * A * speed * speed_rel
        Fm = 0.5 * rho_air * Cl * A * speed**2 * np.cross(omega_hat, v_hat)
        Fnet = Fg + Fd + Fm

        # Update acceleration, velocity, position
        a = Fnet / m
        v += a * dt
        r += v * dt
        traj.append(r.copy())
        t += dt

    return np.array(traj)

# =====================================================
# 4. Run Simulations
# =====================================================
all_trajectories = []

for i in range(num_simulations):
    v0 = np.random.normal(speed_mean, speed_sigma)
    angle = np.random.normal(angle_mean, angle_sigma)
    spin = np.random.normal(spin_mean, spin_sigma)
    traj = simulate_kick(v0, angle, spin)
    all_trajectories.append(traj)

# =====================================================
# 5. Save Final Trajectories as JPEG
# =====================================================
plt.figure(figsize=(12,6))
colors = plt.cm.viridis(np.linspace(0,1,num_simulations))
for i, traj in enumerate(all_trajectories):
    plt.plot(traj[:,0], traj[:,2], color=colors[i])
plt.xlabel('Distance x (m)')
plt.ylabel('Height z (m)')
plt.title('Football Trajectories with Variable Force, Angle, Spin, Wind')
plt.grid(True)
plt.savefig('football_trajectories_wind.jpeg', dpi=300)
plt.show()

# =====================================================
# 6. Animate Trajectories and Save Video
# =====================================================
fig, ax = plt.subplots(figsize=(12,6))
lines = [ax.plot([], [], color=colors[i])[0] for i in range(num_simulations)]
ax.set_xlim(0, max([traj[:,0].max() for traj in all_trajectories])*1.1)
ax.set_ylim(0, max([traj[:,2].max() for traj in all_trajectories])*1.1)
ax.set_xlabel('Distance x (m)')
ax.set_ylabel('Height z (m)')
ax.set_title('Football Trajectories Simulation with Wind Variability')
ax.grid(True)

max_len = max([len(traj) for traj in all_trajectories])

def animate(frame):
    for i, traj in enumerate(all_trajectories):
        if frame < len(traj):
            lines[i].set_data(traj[:frame,0], traj[:frame,2])
        else:
            lines[i].set_data(traj[:,0], traj[:,2])
    return lines

ani = FuncAnimation(fig, animate, frames=max_len, interval=30, blit=False)
writer = FFMpegWriter(fps=30)
ani.save('football_trajectory_simulation_wind.mp4', writer=writer)
plt.close(fig)
