import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters
b = 2.0      # amplitude
k = 1.0      # wave number
omega = 1.0  # angular frequency

# Create x values
x = np.linspace(-2*np.pi, 2*np.pi, 1000)

# Create figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

def animate(frame):
    ax1.clear()
    ax2.clear()
    
    # Calculate current time
    t = frame * 0.1
    
    # Wave displacement
    displacement = b * np.sin(k*x - omega*t)
    
    # Particle velocities
    velocity_x = -b * omega * np.cos(k*x - omega*t)
    
    # Plot 1: Wave with velocity arrows
    ax1.plot(x, displacement, 'b-', linewidth=2, label='Displacement y(x,t)')
    
    # Add velocity arrows
    arrow_positions = np.linspace(-2*np.pi, 2*np.pi, 20)
    for x_pos in arrow_positions:
        y_disp = b * np.sin(k*x_pos - omega*t)
        v_val = -b * omega * np.cos(k*x_pos - omega*t)
        ax1.arrow(x_pos, y_disp, v_val*0.15 , 0, 
                 head_width=0.08, head_length=0.08, fc='red', ec='red', alpha=0.7)
    
    ax1.set_xlim(-2*np.pi, 2*np.pi)
    ax1.set_ylim(-2.5, 2.5)
    ax1.set_ylabel('Displacement y')
    ax1.set_title(f'Traveling Wave with Particle Velocities (t = {t:.2f})')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Velocity field
    ax2.plot(x, velocity_x, 'r-', linewidth=2, label='Particle Velocity v_x(x,t)')
    ax2.set_xlim(-2*np.pi, 2*np.pi)
    ax2.set_ylim(-2.5, 2.5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Velocity v_x')
    ax2.set_title('Particle Velocity Field: v_x = -bω cos(kx - ωt)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    


anim = animation.FuncAnimation(fig, animate, frames=200, interval=50)
plt.tight_layout()
plt.show()
# Save as GIF
anim.save('traveling_wave2.gif', writer='pillow', fps=20)

# Or save as MP4 (requires ffmpeg)
# anim.save('traveling_wave.mp4', writer='ffmpeg', fps=20)
plt.show()