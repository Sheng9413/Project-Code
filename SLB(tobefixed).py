import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N = 50  # number of filaments
eta = 0.001  # viscosity
R_b = 1e-6  # radius of filament head
L = 10e-6  # length of filament
a = 0.1e-6  # radius of filament
b = 0.5e-6  # amplitude
omega = 1.0  # frequency
lambda1 = L/2  # to restrict number of wave to 2
k = 2*np.pi/lambda1  # wave number

# SBT parameter
epsilon = a/L
E = 1 / np.log(2/epsilon)

# Swimming speed from theory
U_theory = -0.5 * b**2 * k * omega

t = 0
# Time parameters
dt = 0.05  # time step (s)
total_time = 5  # total simulation time (s)
num_steps = int(total_time / dt)

x_before = np.linspace(0, L, N)  

x_positions = x_before
y_positions = b * np.sin(k * x_before - omega * t)
xi_par = 2*np.pi*eta/(np.log(2*lambda1/a)-1/2)
xi_per = 2*xi_par
xi_0 = 6*np.pi*eta

# ====== CHANGED: Use theoretical force distribution ======
def get_forces_at_point(x, t):
    """Empirically adjusted for rightward swimming"""
    theta = k*x-omega*t
    xi_para = 2*np.pi*eta/(np.log(2*lambda1/a)-1/2)
    xi_perp = 2*xi_par
    xi_0 = 6*np.pi*eta
    U = - 0.5*k*b**2*omega

    hx = b*k*np.cos(theta)
    ht = -b*omega*np.cos(theta)

    fx = (-xi_para*U
          + (xi_perp-xi_para)*ht*hx) -xi_0*U

    fy = -xi_perp*ht

    return fx, fy

# Precompute forces for all points at current time
def get_all_forces(t):
    """Return arrays of F1 and F2 for all points"""
    F1_array = np.zeros(N)
    F2_array = np.zeros(N)
    for i in range(N):
        f_x, f_y = get_forces_at_point(x_positions[i], t)
        F1_array[i] = f_x
        F2_array[i] = f_y
    return F1_array, F2_array
# =======================================================

def GFU(F1, F2, x1, y1, x, y):
    r1 = (x-x1)
    r2 = y-y1
    r = np.sqrt(r1**2 + r2**2)
    return (r**2*F1 + r1**2*F1 + r1*r2*F2)/(8*np.pi*eta*r**(3)+0.0001) 

def GFV(F1, F2, x1, y1, x, y):
    r1 = (x-x1)
    r2 = y-y1
    r = np.sqrt(r1**2 + r2**2)
    return (r**2*F2 + r1*r2*F1 + r2**2*F2) /(8*np.pi*eta*r**(3)+0.0001) 

trajectories_x = [[] for _ in range(N)]
trajectories_y = [[] for _ in range(N)]

def computevelocityu():
    U_total = []
    # Get forces at current time
    F1_array, F2_array = get_all_forces(t)
    
    for i in range(N):
        x_i = x_positions[i]
        y_i = y_positions[i]
        velocityU = 0
        
        for j in range(N):
            if i == j:
                # Local term uses force at point i
                velocityU += -k*b*np.cos(k*x_i - omega*t)*F2_array[i]/(4*np.pi*eta)
                continue
            x_j = x_positions[j]
            y_j = y_positions[j]
            
            # Stokeslet from point j
            velocityU += GFU(F1_array[j], F2_array[j], x_j, y_j, x_i, y_i) 
            
        U_total.append(velocityU)
        
    return U_total

def computevelocityv():
    V_total = []
    # Get forces at current time
    F1_array, F2_array = get_all_forces(t)
    
    for i in range(N):
        x_i = x_positions[i]
        y_i = y_positions[i]
        velocityV = 0
        
        for j in range(N):
            if i == j:
                # Local term uses force at point i
                velocityV += ( -k*b*np.cos(k*x_i - omega*t)*F1_array[i] + 
                              (1 - b**2*k**2*np.cos(k*x_i - omega*t)**2)*F2_array[i])/(4*np.pi*eta)
                continue
            x_j = x_positions[j]
            y_j = y_positions[j]
            
            # Stokeslet from point j
            velocityV += GFV(F1_array[j], F2_array[j], x_j, y_j, x_i, y_i) 
            
        V_total.append(velocityV)
        
    return V_total

# Store swimming speed estimates
swimming_speeds = []

while t < 20:
    # Store trajectories
    for i in range(N):
        trajectories_x[i].append(x_positions[i])
        trajectories_y[i].append(y_positions[i])
    
    # Compute velocities
    U_all = computevelocityu()
    V_all = computevelocityv()
    
    # Update positions
    for i in range(N):
        x_positions[i] = x_positions[i] + dt * U_all[i]
        y_positions[i] = y_positions[i] + dt * V_all[i]

    
    # Estimate swimming speed from center of mass
    if len(trajectories_x[0]) > 1:
        current_frame = len(trajectories_x[0]) - 1
        prev_frame = current_frame - 1
        com_current = np.mean([trajectories_x[i][current_frame] for i in range(N)])
        com_prev = np.mean([trajectories_x[i][prev_frame] for i in range(N)])
        U_est = (com_current - com_prev) / dt
        swimming_speeds.append(U_est)
    
    t = t + dt

X_trajectories = np.array(trajectories_x)  
Y_trajectories = np.array(trajectories_y)

# Analysis
print(f"Theoretical swimming speed: {U_theory:.3e} m/s = {U_theory*1e6:.2f} µm/s")
if swimming_speeds:
    avg_speed = np.mean(swimming_speeds)
    print(f"Simulated average speed: {avg_speed:.3e} m/s = {avg_speed*1e6:.2f} µm/s")
    print(f"Ratio (sim/theory): {avg_speed/U_theory:.3f}")

# Check force balance at final time
F1_final, F2_final = get_all_forces(t)
total_Fx = np.sum(F1_final) * (L/(N-1))
total_Fy = np.sum(F2_final) * (L/(N-1))
print(f"Total force: Fx = {total_Fx:.2e} N, Fy = {total_Fy:.2e} N")

# Animation (your original code)
all_x = X_trajectories.flatten()
all_y = Y_trajectories.flatten()
x_min, x_max = np.min(all_x)*1e6, np.max(all_x)*1e6
y_min, y_max = np.min(all_y)*1e6, np.max(all_y)*1e6

margin_x = (x_max - x_min) * 0.1
margin_y = (y_max - y_min) * 0.1

fig_simple, ax_simple = plt.subplots(figsize=(10, 6))

scatters_simple = []
for i in range(N):
    scat = ax_simple.scatter([], [], s=40, c='red', alpha=0.7, 
                            edgecolors='black', linewidth=0.5)
    scatters_simple.append(scat)

ax_simple.set_xlim(x_min - margin_x, x_max + margin_x)
ax_simple.set_ylim(y_min - margin_y, y_max + margin_y)
ax_simple.set_xlabel('x Position (µm)')
ax_simple.set_ylabel('y Displacement (µm)')
ax_simple.grid(True, alpha=0.3)
ax_simple.axhline(y=0, color='k', linestyle=':', alpha=0.5)

def animate_simple(frame):
    for i in range(N):
        scatters_simple[i].set_offsets(
            np.array([[X_trajectories[i, frame] * 1e6, 
                      Y_trajectories[i, frame] * 1e6]])
        )

    return scatters_simple

actual_frames = len(trajectories_x[0])
max_frames = min(200, actual_frames)
frames_to_show = np.linspace(0, actual_frames-1, max_frames, dtype=int)

ani = animation.FuncAnimation(fig_simple, animate_simple, 
                             frames=frames_to_show,
                             interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()
ani.save('SBT24.gif', writer=animation.PillowWriter(fps=20, bitrate=1800))