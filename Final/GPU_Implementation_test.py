import numpy as np
import matplotlib.pyplot as plt
from Volterra_In_Out_6 import VolterraResponse
import time
from datetime import datetime

def main():
    h0 = 0.0
    h1 = np.zeros(shape=(30))
    h2 = np.zeros(shape=(30, 30))
    h3 = np.zeros(shape=(30, 30, 30))
    h4 = np.zeros(shape=(30, 30, 30, 30))
    h5 = np.zeros(shape=(30, 30, 30, 30, 30))

    # h1[1] = 1.0
    # h1[8] = 1.0
    # h1[6] = 5.0
    # h2[1, 2] = 1.0
    # h2[8,2] = 2.0
    # h2[8, 8] = 2.0
    # h3[1, 2, 3] = 1.0
    # h3[8, 8, 8] = 3.0
    # h3[1, 2, 8] = 1.0

    # h4[1, 2, 3, 4] = 40.0
    # h4[8,8,8,8] = 33.0
    # h4[1, 2, 3, 8] = 1.0
    # h4[3,2,5,1] = 2.0
    # h5[3,3,3,3,3] = 2.0
    # h5[1, 2, 3, 4, 5] = 1.0
    # h5[3,1,8,4,1] = 1.0
    # h5[1, 2, 3, 4, 8] = 50


    h1[:6] = np.random.rand(6).astype(np.float32)
    h2[:6, :6] = np.random.rand(6, 6).astype(np.float32)
    h3[:6, :6, :6] = np.random.rand(6, 6, 6).astype(np.float32)
    h4[:6, :6, :6, :6] = np.random.rand(6, 6, 6, 6).astype(np.float32)
    h5[:6, :6, :6, :6, :6] = np.random.rand(6, 6, 6, 6, 6).astype(np.float32)



    input_len = 1000
    x = np.sin(np.linspace(0, 2*np.pi, input_len)).astype(np.float32) +1
    x = np.random.rand(input_len).astype(np.float32)

    # Create Volterra system instances for CPU and GPU
    volterra_cpu = VolterraResponse(use_opencl=False)
    volterra_gpu = VolterraResponse(use_opencl=True)

    volterra_cpu.set_kernels(h0, h1, h2, h3, h4, h5)
    volterra_gpu.set_kernels(h0, h1, h2, h3, h4, h5)

    # Measure computation time for CPU
    start_time_cpu = time.time()
    y_cpu = volterra_cpu.Volterra_In_Out(x)
    cpu_time = time.time() - start_time_cpu

    # Measure computation time for GPU
    start_time_gpu = time.time()
    y_gpu, profiling_data = volterra_gpu.Volterra_In_Out(x, profiling_data=True)
    gpu_time = time.time() - start_time_gpu

    # Calculate the difference and relative error between GPU and CPU outputs
    diff = y_gpu - y_cpu
    relative_error = diff/y_cpu * 100  # Convert to percent

    err_250 = (y_gpu[250] - y_cpu[250]) / y_cpu[250] * 100
    print(f"Relative error at index 250: {err_250:.2f}%")

    print(y_cpu.shape, y_gpu.shape, diff.shape, relative_error.shape)
    # Calculate speedup
    speedup = cpu_time / gpu_time

    # Plot the output of the Volterra system for both CPU and GPU
    plt.figure(figsize=(12, 6))
    plt.plot(y_cpu, label='CPU Output', linestyle='--')
    plt.plot(y_gpu, label='GPU Output', linestyle='-.')
    plt.plot(diff, label='Difference', linestyle=':')
    plt.title(f'Volterra System Output Comparison\nInput Length: {input_len}')
    plt.xlabel('Sample Index')
    plt.ylabel('Output')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plot_filename = f'volterra_output_comparison_input_{input_len}.png'
    plt.savefig(plot_filename)

    # Display the plot
    plt.show()
    plt.close()

    # Plot the relative error
    plt.figure(figsize=(12, 6))
    plt.plot(relative_error, label='Relative Error (%)', color='red')
    plt.title(f'Relative Error between GPU and CPU Outputs\nInput Length: {input_len}')
    plt.xlabel('Sample Index')
    plt.ylabel('Relative Error %')
    plt.legend()
    plt.grid(True)

    # Save the relative error plot as a PNG file
    error_plot_filename = f'volterra_relative_error_input_{input_len}.png'
    plt.savefig(error_plot_filename)

    # Display the relative error plot
    plt.show()
    plt.close()

    # Print computation times
    print(f"Input Length: {input_len}")
    print(f"CPU computation time: {cpu_time:.6f} seconds")
    print(f"GPU computation time: {gpu_time:.6f} seconds")
    print(f"Speedup: {speedup:.2f}x")

    # Print GPU profiling data
    print("GPU Profiling Data:")
    print(profiling_data)

if __name__ == '__main__':
    main()
