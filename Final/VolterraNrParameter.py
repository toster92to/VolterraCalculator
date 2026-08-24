from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb


def volterra_parameters(order, kernel_length):
    # Intelligente Verfahren: Identifikation und Regelung nichtlinearer Systeme p 261
    return comb(kernel_length + order - 1, order, exact=True)


def cumulative_parameters(max_order, kernel_length):
    cumulative_params = []
    total_params = 0
    for order in range(1, max_order + 1):
        params = volterra_parameters(order, kernel_length)
        total_params += params
        cumulative_params.append(total_params)
    return cumulative_params


kernel_lengths = [10, 20, 30]
max_order = 10

plt.figure(figsize=(12, 10))

colors = ['darkblue', 'blue', 'darkgray']
markers = ['o', 'o', 'o']

for idx, length in enumerate(kernel_lengths):
    orders = list(range(1, max_order + 1))
    parameters = cumulative_parameters(max_order, length)
    plt.plot(orders, parameters, marker=markers[idx], linestyle='-', color=colors[idx],
             markersize=8, linewidth=2, label=f"Kernel Length = {length}")


plt.yscale('log')

plt.xlabel('Volterra Series Order', fontsize=29)
plt.ylabel('Cumulative Number of Parameters', fontsize=29)

plt.title('Cumulative Number of Parameters in Volterra Series vs. Order for Various Kernel Lengths',
          fontsize=30).set_position([0.5, 0.04])

plt.grid(True, which="both", linestyle='--', linewidth=0.5)
plt.xticks(orders, fontsize=23)
plt.yticks(fontsize=23)
plt.legend(fontsize=23)
plt.tight_layout()

plt.show()

##################################################################################################################


x = np.linspace(0, 10, 100)
y_a = np.exp(-0.1*x) * np.sin(2*np.pi*x)
y_b = 0.6*np.exp(-0.2*x) * np.sin(2*np.pi*0.6*x)
y_c = 0.4*np.exp(-0.25*x) * np.cos(2*np.pi*0.8*x)

plt.figure(figsize=(10, 6))

plt.plot(x, y_a, label='(a)', color='black', linewidth=2)
plt.plot(x, y_b, label='(b)', color='gray', linewidth=2)
plt.plot(x, y_c, label='(c)', color='orange', linestyle='--', linewidth=2)

plt.scatter([2, 5, 8], [y_a[20], y_b[50], y_c[80]], color='orange', zorder=5)

plt.xlabel('f', fontsize=12)
plt.ylabel('F(y)', fontsize=12)
plt.title('Example Plot Replication', fontsize=14)
plt.legend(title='Legend')

plt.grid(True, linestyle='--', alpha=0.5)

plt.show()

##################################################################################################################
# Convolution operation


# Define the input signal and kernel
input_signal = np.array([1, 2, 3, 4, 5])
kernel = np.array([1, 0.5])

# Compute the convolution
convolution_output = np.convolve(input_signal, kernel, 'full')

# Create a series of plots
for i in range(len(convolution_output)):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    ax1.set_ylim(0, np.max(input_signal) + 1)
    ax2.set_ylim(0, np.max(convolution_output) + 1)

    # Plot the input signal and kernel
    ax1.plot(input_signal, 'b-', label='Input Signal')
    ax1.plot(np.arange(i, i + len(kernel)), kernel, 'g-', label='Kernel')
    ax1.legend()

    # Plot the convolution output up to the current position of the kernel
    ax2.plot(convolution_output[:i+1], 'r-', label='Convolution Output')
    ax2.legend()

    # Display the plot
    plt.show()
