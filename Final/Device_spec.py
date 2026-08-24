import pyopencl as cl
import subprocess
import shutil

import matplotlib.pyplot as plt

def get_cpu_info():
    try:
        result = subprocess.run(['lscpu'], stdout=subprocess.PIPE, check=True)
        return result.stdout.decode('utf-8')
    except subprocess.CalledProcessError as e:
        return f"Error getting CPU info: {e}"

def get_gpu_info():
    if shutil.which('lspci'):
        try:
            result = subprocess.run(['lspci'], stdout=subprocess.PIPE, check=True)
            gpu_info = [line for line in result.stdout.decode('utf-8').split('\n') if 'VGA' in line or '3D' in line]
            return '\n'.join(gpu_info)
        except subprocess.CalledProcessError as e:
            return f"Error getting GPU info: {e}"
    else:
        return "lspci command not found. Please install it using 'sudo apt install pciutils'."

def get_detailed_system_info():
    if shutil.which('lshw'):
        try:
            result = subprocess.run(['sudo', 'lshw', '-short'], stdout=subprocess.PIPE, check=True)
            return result.stdout.decode('utf-8')
        except subprocess.CalledProcessError as e:
            return f"Error getting detailed system info: {e}"
    else:
        return "lshw command not found. Please install it using 'sudo apt install lshw'."

def main():
    print("CPU Information:")
    print(get_cpu_info())

    print("\nGPU Information:")
    print(get_gpu_info())

    print("\nDetailed System Information:")
    print(get_detailed_system_info())

    print("\nOpenCL Platforms and Devices:")

    # Get the list of available OpenCL platforms
    platforms = cl.get_platforms()

    # Check if any platforms were found
    if not platforms:
        print("No OpenCL platforms found.")
    else:
        # Iterate over the platforms
        for platform in platforms:
            print(f"Platform: {platform.name}, Version: {platform.version}, Vendor: {platform.vendor}")

            # Try to get the list of devices available on this platform
            try:
                devices = platform.get_devices()
                if not devices:
                    print("  No devices found.")
                else:
                    # Iterate over the devices
                    for device in devices:
                        # Print device details
                        device_type = cl.device_type.to_string(device.type)
                        print(f"  Device: {device.name}, Type: {device_type}, Version: {device.version}, Vendor: {device.vendor}")
            except cl.RuntimeError as e:
                print(f"  Failed to get devices for platform {platform.name}: {e}")


if __name__ == "__main__":
    main()
