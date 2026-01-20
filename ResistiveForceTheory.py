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

# drag coefficient
xi_par = 2*np.pi*eta/(np.log(2*lambda1/a)-1/2)
xi_per = 2*xi_par
xi_0 = 6*np.pi*eta

# average velocity
U_avg = -0.5 *xi_par* omega * k * b**2 / (1 + (xi_0 * R_b) / (xi_par * L))

x_before = np.linspace(0, L, N)  

trajectories_x = [[] for _ in range(N)]  # list of N lists
trajectories_y = [[] for _ in range(N)]  

t = 0
dt = 0.05  # time step
total_time = 10 
num_steps = int(total_time / dt)

x_positions = x_before
y_positions = b * np.sin(k * x_before - omega * t)

for step in range(num_steps):
    for i in range(N):
        trajectories_x[i].append(x_positions[i])
        trajectories_y[i].append(y_positions[i])

    t += dt
    
    for i in range(N):
        #euler time stepping
        x_after = x_before[i] + U_avg * t
        y_after = b * np.sin(k * x_after - omega * t)
        
        x_positions[i] = x_after
        y_positions[i] = y_after
    


X_trajectories = np.array(trajectories_x)  
Y_trajectories = np.array(trajectories_y)  


# to find the min max x,y value in trajextory
all_x = X_trajectories.flatten()
all_y = Y_trajectories.flatten()
x_min, x_max = np.min(all_x)*1e6, np.max(all_x)*1e6
y_min, y_max = np.min(all_y)*1e6, np.max(all_y)*1e6

margin_x = (x_max - x_min) * 0.1
margin_y = (y_max - y_min) * 0.1


fig_simple, ax_simple = plt.subplots(figsize=(10, 6))

# create scatter plot
scatters_simple = []
for i in range(N):
    scat = ax_simple.scatter([], [], s=40, c='red', alpha=0.7, 
                            edgecolors='black', linewidth=0.5)
    scatters_simple.append(scat)


ax_simple.set_xlim(x_min - margin_x, x_max + margin_x)
ax_simple.set_ylim(y_min - margin_y, y_max + margin_y)
ax_simple.set_xlabel('x Position (µm)')
ax_simple.set_ylabel('y Displacement (µm)')
ax_simple.set_title(f'{N} Filament Points Swimming')
ax_simple.grid(True, alpha=0.3)
ax_simple.axhline(y=0, color='k', linestyle=':', alpha=0.5)


def animate_simple(frame):
    for i in range(N):
        scatters_simple[i].set_offsets(
            np.array([[X_trajectories[i, frame] * 1e6, 
                      Y_trajectories[i, frame] * 1e6]])
        )
    return scatters_simple

max_frames = min(200, num_steps)
frames_to_show = np.linspace(0, num_steps-1, max_frames, dtype=int)

ani = animation.FuncAnimation(fig_simple, animate_simple,frames=frames_to_show,interval=50,blit=True,repeat=True)

plt.tight_layout()
plt.show()
ani.save('RFT4.gif', writer=animation.PillowWriter(fps=20, bitrate=1800))
