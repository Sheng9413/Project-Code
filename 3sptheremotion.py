import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation

# Greens Function
def GFU(F1,F2,x1,y1,x,y,a):
    r1 = (x-x1)
    r2 = y-y1
    r = np.sqrt( r1**2+r2**2)
    return (r**2*F1 + r1**2*F1+r1*r2*F2)/(r**(3)+0.0001) + a**2/6 * ((2*r**2-6*r1**2)*F1 - 6*r1*r2*F2 )/(r**(5)+0.0001)

def GFV(F1,F2,x1,y1,x,y,a):
    r1 = (x-x1)
    r2 = y-y1
    r = np.sqrt( r1**2+r2**2)
    return ( r**2*F2+r1*r2*F1+r2**2*F2) /(r**(3)+0.0001) + a**2/6 * ((-6*r1*r2*F1) + (2*r**2-6*r2**2)*F2)/(r**(5)+0.0001)



# Initial position
x1 = -5
y1 = 1

x2 = 0
y2 = 1

x3 = 7
y3 = 1

a=1


t=0
#velocity for each particle
U1 = GFU(0,-1,x2,y2,x1,y1,a)  + GFU(0,-1,x3,y3,x1,y1,a)
V1 = GFV(0,-1,x2,y2,x1,y1,a)  + GFV(0,-1,x3,y3,x1,y1,a)

U2 = GFU(0,-1,x1,y1,x2,y2,a)  + GFU(0,-1,x3,y3,x2,y2,a)
V2 = GFV(0,-1,x1,y1,x2,y2,a)  + GFV(0,-1,x3,y3,x2,y2,a)

U3 = GFU(0,-1,x2,y2,x3,y3,a)  + GFU(0,-1,x1,y1,x3,y3,a)
V3 = GFV(0,-1,x2,y2,x3,y3,a)  + GFV(0,-1,x1,y1,x3,y3,a)


trajectories_x1 = []
trajectories_y1 = []

trajectories_x2 = []
trajectories_y2 = []

trajectories_x3 = []
trajectories_y3 = []



t=0
dt=0.5

while t <200: 
    x1 = x1 + dt*U1
    y1 = y1 + dt*V1
    trajectories_x1.append(x1)
    trajectories_y1.append(y1)
    
    x2 = x2 + dt*U2
    y2 = y2 + dt*V2
    trajectories_x2.append(x2)
    trajectories_y2.append(y2)
    
    x3 = x3 + dt*U3
    y3 = y3 + dt*V3
    trajectories_x3.append(x3)
    trajectories_y3.append(y3)

    
    U1 = GFU(0,-1,x2,y2,x1,y1,a)  + GFU(0,-1,x3,y3,x1,y1,a)
    V1 = GFV(0,-1,x2,y2,x1,y1,a)  + GFV(0,-1,x3,y3,x1,y1,a)

    U2 = GFU(0,-1,x1,y1,x2,y2,a)  + GFU(0,-1,x3,y3,x2,y2,a)
    V2 = GFV(0,-1,x1,y1,x2,y2,a)  + GFV(0,-1,x3,y3,x2,y2,a)

    U3 = GFU(0,-1,x2,y2,x3,y3,a)  + GFU(0,-1,x1,y1,x3,y3,a)
    V3 = GFV(0,-1,x2,y2,x3,y3,a)  + GFV(0,-1,x1,y1,x3,y3,a)

    t=t+dt

X1 = np.array(trajectories_x1)
Y1 = np.array(trajectories_y1)
X2 = np.array(trajectories_x2)
Y2 = np.array(trajectories_y2)
X3 = np.array(trajectories_x3)
Y3 = np.array(trajectories_y3)



fig, ax = plt.subplots()
ax.set_xlim([-10, 10])
ax.set_ylim([-100, 1])

# Create three separate scatter plots with different colors
scat1 = ax.scatter([], [], c='red', s=50, label='Particle 1')    
scat2 = ax.scatter([], [], c='blue', s=50, label='Particle 2')    
scat3 = ax.scatter([], [], c='green', s=50, label='Particle 3')


ax.legend()

def animate(i):
    # Update each scatter plot
    scat1.set_offsets(np.array([[X1[i], Y1[i]]]))
    scat2.set_offsets(np.array([[X2[i], Y2[i]]]))
    scat3.set_offsets(np.array([[X3[i], Y3[i]]]))
    
    return (scat1, scat2, scat3)

ani = animation.FuncAnimation(fig, animate, repeat=True, frames=len(X1) - 1, interval=50, blit=True)

writer = animation.PillowWriter(fps=15,
                                metadata=dict(artist='Me'),
                                bitrate=1800)
ani.save('3sphere.gif', writer=writer)

plt.show()