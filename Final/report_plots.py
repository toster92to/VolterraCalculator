import json
import pandas as pd
import matplotlib.pyplot as plt
import os

input_file_volcomp = "Final/volcomp_reports/0612_1112_volcomp.json"
input_file_volterra = "Final/GPUvsCPU_reports/0612_1120GPUvsCPU.json"

# Volterra comparison
# Function to plot  metrics
def plot_metrics(df, metric, y_label, title, save_path=None):
    plt.figure(figsize=(12, 6))
    for kernel in df['kernel_h1'].unique():
        subset = df[df['kernel_h1'] == kernel]
        plt.plot(subset['input_size'], subset[metric],
                 marker='o', label=f'Kernel Length {kernel}')
    plt.xscale('log')
    plt.yscale('linear')
    plt.xlabel('Input Size')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# Extract base names for image file naming
volcomp_base_name = os.path.splitext(os.path.basename(input_file_volcomp))[0]
volterra_base_name = os.path.splitext(os.path.basename(input_file_volterra))[0]

# Load volcomp data
with open(input_file_volcomp, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Sort by timestamp and drop duplicates, keeping the most recent entry
df_sorted = df.sort_values(by='timestamp').drop_duplicates(
    subset=['implementation', 'input_size'], keep='last')

# Extract  relevant columns
df_filtered = df_sorted[['implementation', 'input_size', 'gpu_time']]

df_pivot = df_filtered.pivot(
    index='input_size', columns='implementation', values='gpu_time')

plt.figure(figsize=(10, 6))
for implementation in df_pivot.columns:
    plt.plot(df_pivot.index,
             df_pivot[implementation], marker='o', label=implementation)

plt.xlabel('Input Size')
plt.ylabel('GPU Time (s)')
plt.title('GPU Computation Time for Different Input Sizes and Implementations')
plt.legend()
plt.grid(True)
plt.savefig(f"Final/volcomp_reports/{volcomp_base_name}.png")
plt.show()

############################################################################################################
# GPU vs CPU comparisson

with open(input_file_volterra, "r") as f:
    data = json.load(f)

df = pd.DataFrame(data)

# Extracting the kernel lengths
df['kernel_h1'] = df['kernel_lengths'].apply(lambda x: x['h1'])
df['kernel_h2'] = df['kernel_lengths'].apply(lambda x: x['h2'][0])
df['kernel_h3'] = df['kernel_lengths'].apply(lambda x: x['h3'][0])
df['kernel_h4'] = df['kernel_lengths'].apply(lambda x: x['h4'][0])
df['kernel_h5'] = df['kernel_lengths'].apply(lambda x: x['h5'][0])

# Define the folder for saving the plots
output_folder_volterra = os.path.dirname(input_file_volterra)
os.makedirs(output_folder_volterra, exist_ok=True)

# Plotting Speedup vs. Input Size
plot_metrics(df, 'speedup', 'Speedup (CPU Time / GPU Time)',
             'Speedup Factor for Different Kernel and Input Sizes',
             os.path.join(output_folder_volterra, f'{volterra_base_name}_Speedup.png'))

# Plotting CPU Time vs. Input Size
plot_metrics(df, 'cpu_time', 'CPU Time (seconds)',
             'CPU Time for Different Kernel and Input Sizes',
             os.path.join(output_folder_volterra, f'{volterra_base_name}_CPU.png'))

# Plotting GPU Time vs. Input Size
plot_metrics(df, 'gpu_time', 'GPU Time (seconds)',
             'GPU Time for Different Kernel and Input Sizes',
             os.path.join(output_folder_volterra, f'{volterra_base_name}_GPU.png'))

# Plotting Queuing Overhead vs. Input Size (display only)
for i in range(1, 6):
    df[f'queuing_overhead_H{i}'] = df['gpu_profiling_data'].apply(
        lambda x: x[f'H{i}']['queuing_overhead (us)'])
    plot_metrics(df, f'queuing_overhead_H{i}', 'Queuing Overhead (us)',
                 f'Queuing Overhead for H{i} vs. Input Size')

# Plotting Device Latency vs. Input Size (display only)
for i in range(1, 6):
    df[f'device_latency_H{i}'] = df['gpu_profiling_data'].apply(
        lambda x: x[f'H{i}']['device_latency (us)'])
    plot_metrics(df, f'device_latency_H{i}', 'Device Latency (us)',
                 f'Device Latency for H{i} vs. Input Size')

# Plotting Execution Time vs. Input Size (display only)
for i in range(1, 6):
    df[f'execution_time_H{i}'] = df['gpu_profiling_data'].apply(
        lambda x: x[f'H{i}']['execution_time (s)'])
    plot_metrics(df, f'execution_time_H{i}', 'Execution Time (seconds)',
                 f'Execution Time for H{i} vs. Input Size')
