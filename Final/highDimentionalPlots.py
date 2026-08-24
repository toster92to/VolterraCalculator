import numpy as np
import matplotlib.pyplot as plt

# Example kernel function h_3(t1, t2, t3)


def h_3(t1, t2, t3):
    return np.exp(-0.5*(t1**2 + t2**2 + t3**2))


# Generate a grid of t1, t2 values
t1, t2 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
fixed_t3 = 0.5  # Fix t3

# Compute the kernel values
values = h_3(t1, t2, fixed_t3)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(t1, t2, values, levels=50, cmap='viridis')
plt.colorbar()
plt.title('Third-order Volterra Kernel Slice at t3=0.5')
plt.xlabel('t1')
plt.ylabel('t2')
plt.show()
