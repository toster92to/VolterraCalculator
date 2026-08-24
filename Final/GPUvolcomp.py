import importlib
import os
import json
import numpy as np
import time
from datetime import datetime
import matplotlib.pyplot as plt

# Fcreate a unique report file name with date and time
def get_report_file():
    now = datetime.now()
    report_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'volcomp_reports')
    os.makedirs(report_dir, exist_ok=True)
    return os.path.join(report_dir, f"{now.strftime('%m%d_%H%M')}_volcomp.json")

def update_report(report_file, data):
    if os.path.exists(report_file):
        with open(report_file, 'r') as file:
            report = json.load(file)
    else:
        report = []

    report.append(data)

    with open(report_file, 'w') as file:
        json.dump(report, file, indent=2, separators=(',', ': '))

# Function to benchmark an implementation
def benchmark_implementation(module_name, report_file):
    module = importlib.import_module(module_name)
    VolterraResponse = module.VolterraResponse

    np.random.seed(0)

    kernel_lengths = (10, 20, 30, 40)
    input_lengths = [1000 * 2**i for i in range(0, 15)]

    for kernel_len in kernel_lengths:
        for input_len in input_lengths:
            h0 = 1
            h1 = np.random.rand(kernel_len).astype(np.float32)
            h2 = np.random.rand(kernel_len, kernel_len).astype(np.float32)
            h3 = np.random.rand(kernel_len, kernel_len, kernel_len).astype(np.float32)
            h4 = np.random.rand(kernel_len, kernel_len, kernel_len, kernel_len).astype(np.float32)
            h5 = np.random.rand(kernel_len, kernel_len, kernel_len, kernel_len, kernel_len).astype(np.float32)
            x = np.random.rand(input_len).astype(np.float32)

            volterra_gpu = VolterraResponse(use_opencl=True)
            volterra_gpu.set_kernels(h0, h1, h2, h3, h4, h5)

            start_time_gpu = time.time()
            y_gpu, profiling_data = volterra_gpu.Volterra_In_Out(x, return_profiling_data=True)
            gpu_time = time.time() - start_time_gpu

            report_data = {
                "implementation": module_name,
                "timestamp": datetime.now().isoformat(),
                "input_size": len(x),
                "kernel_lengths": {
                    "h1": len(h1),
                    "h2": h2.shape,
                    "h3": h3.shape,
                    "h4": h4.shape,
                    "h5": h5.shape,
                },
                "gpu_time": gpu_time,
                "gpu_profiling_data": profiling_data,
            }

            update_report(report_file, report_data)
            print(f"Implementation: {module_name}, Kernel Length: {kernel_len}, Input Length: {input_len}")
            print(f"GPU computation time: {gpu_time:.6f} seconds")

# unique report file for thhis run
report_file = get_report_file()

# mplementation modules
implementations = [
    "Volterra_In_Out_5",
    "Volterra_In_Out_6",
]

for impl in implementations:
    benchmark_implementation(impl, report_file)

