from scipy.optimize import curve_fit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from Volterra_In_Out_5 import VolterraResponse
import os
import time
from datetime import datetime


def get_report_file():
    now = datetime.now()
    report_dir = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), 'GPUvsCPU_reports')
    os.makedirs(report_dir, exist_ok=True)
    return os.path.join(report_dir, f"{now.strftime('%m%d_%H%M')}GPUvsCPU.json")


def update_report(report_file, data):
    if os.path.exists(report_file):
        with open(report_file, 'r') as file:
            report = json.load(file)
    else:
        report = []

    report.append(data)

    with open(report_file, 'w') as file:
        json.dump(report, file, indent=2, separators=(',', ': '))


report_file = get_report_file()

np.random.seed(0)

kernel_lengths = range(10, 110, 10)
input_lengths = [1000 * 2**i for i in range(0, 14)]  # 1000, 2000, 4000, ...

for kernel_len in kernel_lengths:
    for input_len in input_lengths:
        # Create random kernels of the current length
        h0 = 1
        h1 = np.random.rand(kernel_len).astype(np.float32)
        h2 = np.random.rand(kernel_len, kernel_len).astype(np.float32)
        h3 = np.random.rand(kernel_len, kernel_len,
                            kernel_len).astype(np.float32)
        h4 = np.random.rand(kernel_len, kernel_len,
                            kernel_len, kernel_len).astype(np.float32)
        h5 = np.random.rand(kernel_len, kernel_len, kernel_len,
                            kernel_len, kernel_len).astype(np.float32)

        x = np.random.rand(input_len).astype(np.float32)

        #  Volterra system instances for CPU and GPU
        volterra_cpu = VolterraResponse(use_opencl=False)
        volterra_gpu = VolterraResponse(use_opencl=True)

        volterra_cpu.set_kernels(h0, h1, h2, h3, h4, h5)
        volterra_gpu.set_kernels(h0, h1, h2, h3, h4, h5)

        start_time_cpu = time.time()
        y_cpu, _ = volterra_cpu.Volterra_In_Out(x)
        cpu_time = time.time() - start_time_cpu

        start_time_gpu = time.time()
        y_gpu, profiling_data = volterra_gpu.Volterra_In_Out(x)
        gpu_time = time.time() - start_time_gpu

        speedup = cpu_time / gpu_time

        # Prepare data for report
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "input_size": len(x),
            "kernel_lengths": {
                "h1": len(h1),
                "h2": h2.shape,
                "h3": h3.shape,
                "h4": h4.shape,
                "h5": h5.shape,
            },
            "cpu_time": cpu_time,
            "gpu_time": gpu_time,
            "speedup": speedup,
            "gpu_profiling_data": profiling_data,
        }

        # Update the report file
        update_report(report_file, report_data)

        # Print computation times
        print(f"Kernel Length: {kernel_len}, Input Length: {input_len}")
        print(f"CPU computation time: {cpu_time:.6f} seconds")
        print(f"GPU computation time: {gpu_time:.6f} seconds")
        print(f"Speedup: {speedup:.2f}x")
