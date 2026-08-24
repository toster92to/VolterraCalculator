import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, permutations
import plotly.graph_objects as go
from Volterra_In_Out_6 import VolterraResponse
from Volterra_Operations import make_symmetrical

# Number of points
n = 300

# Define the input signals
x1 = np.zeros(n)  # Dirac delta function
x1[n // 2] = 1
x2 = np.concatenate((np.zeros(n // 2), np.ones(n // 2)))  # Step function
x3 = np.linspace(0, 1, n)  # Ramp function
x4 = np.sin(np.linspace(0, 2 * np.pi, n))  # Sin wave
x5 = np.sign(np.sin(np.linspace(0, 2 * np.pi, n)))  # Square wave
x6 = np.random.normal(0, 1, n)  # White noise

# Plot all signals
signals = [x1, x2, x3, x4, x5, x6]
titles = ['Dirac Delta', 'Step', 'Ramp', 'Sin', 'Square', 'White Noise']

# Define the input signal
x = x4

# Define delayed versions of the input signal
x_delay1 = np.concatenate(([0], x[:-1]))
x_delay2 = np.concatenate(([0, 0], x[:-2]))
x_delay3 = np.concatenate(([0, 0, 0], x[:-3]))
x_delay4 = np.concatenate(([0, 0, 0, 0], x[:-4]))

# Calculate the dynamic system output using the given kernels
y_dynamic = (
    30 * x -
    6 * x_delay1 * x_delay3 +
    3 * x_delay1 * x * x_delay4 -
    x ** 4 +
    0.1 * x_delay2 * x ** 3 * x_delay1
)

# Initialize the Volterra response system
v = VolterraResponse(use_opencl=False)
v.h1[0] = 30
v.h2[1, 3] = -6
v.h3[0, 1, 4] = 3
v.h4[0, 0, 0, 0] = -1
# v.h5[0, 0, 0, 1, 2] = 0.1

v = make_symmetrical(v)

# Calculate the Volterra system output
v_out = v.Volterra_In_Out(x)
v_real = y_dynamic

# Plot the results
plt.figure()
plt.plot(v_out, label='Volterra')
plt.plot(v_real, label='Real')
plt.legend()
plt.title('Comparison of Volterra and Real System Outputs')
plt.xlabel('Time')
plt.ylabel('Output')
plt.show()
