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


t=0
# Time parameters
dt = 0.05  # time step (s)
total_time = 5  # total simulation time (s)
num_steps = int(total_time / dt)

x_before = np.linspace(0, L, N)  

x_positions = x_before
y_positions = b * np.sin(k * x_before - omega * t)

#force in x component #not sure what forces to put down
F1 = 0

#force in y component to the fluid
F2 = 0


def GFU(F1,F2,x1,y1,x,y):
    r1 = (x-x1)
    r2 = y-y1
    r = np.sqrt( r1**2+r2**2)
    return (r**2*F1 + r1**2*F1+r1*r2*F2)/(8*np.pi*eta*r**(3)+0.0001) 

def GFV(F1,F2,x1,y1,x,y):
    r1 = (x-x1)
    r2 = y-y1
    r = np.sqrt( r1**2+r2**2)
    return ( r**2*F2+r1*r2*F1+r2**2*F2) /(8*np.pi*eta*r**(3)+0.0001) 

trajectories_x = [[] for _ in range(N)]  # list of N lists
trajectories_y = [[] for _ in range(N)]  

velocities = np.zeros(N)

def computevelocityu(x,y):
    U_total = []
    for i in range(N):
        x_i = x_positions[i]
        y_i = y_positions[i]
        velocityU = 0
        
        for j in range(N):
            if i == j:
                velocityU +=  -k*b*np.cos(k*x_i-omega*t)*F2/(4*np.pi*eta)
                continue
            x_j = x_positions[j]
            y_j = y_positions[j]
            
            
            velocityU += GFU(F1,F2,x_j,y_j,x_i,y_i) 
            
        U_total.append(velocityU)
        
    return U_total

def computevelocityv(x,y):
    V_total = []
    for i in range(N):
        x_i = x_positions[i]
        y_i = y_positions[i]
        velocityV = 0
        
        for j in range(N):
            if i == j:
                velocityV += ( -k*b*np.cos(k*x_i-omega*t)*F1+ (1-b**2*k**2*np.cos(k*x_i-omega*t)**2)*F2)/(4*np.pi*eta)
                continue
            x_j = x_positions[j]
            y_j = y_positions[j]
            
            
            velocityV += GFV(F1,F2,x_j,y_j,x_i,y_i) 
            
        V_total.append(velocityV)
        
    return V_total
        
while t < 5:
    
    for i in range(N):
        trajectories_x[i].append(x_positions[i])
        trajectories_y[i].append(y_positions[i])
    

    U_all = computevelocityu(x_positions,y_positions)  
    V_all = computevelocityv(x_positions,y_positions)  
    

    for i in range(N):
        x_positions[i] = x_positions[i] + dt * U_all[i]
        y_positions[i] = y_positions[i] + dt * V_all[i]
    

    for i in range(N):
        y_positions[i] = b * np.sin(k * x_positions[i] - omega * (t + dt))
    
    t = t + dt
    
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



actual_frames = len(trajectories_x[0])  
max_frames = min(200, actual_frames)
frames_to_show = np.linspace(0, actual_frames-1, max_frames, dtype=int)

ani = animation.FuncAnimation(fig_simple, animate_simple, 
                             frames=frames_to_show,  
                             interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()
ani.save('SBT1.gif', writer=animation.PillowWriter(fps=20, bitrate=1800))


