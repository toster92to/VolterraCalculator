import pyopencl as cl
import os

# Set environment variables for OpenCL
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'
os.environ['PYOPENCL_CTX'] = '1'
os.environ['PYOPENCL_NO_CACHE'] = '1' 

def list_platforms_and_devices():
    """Lists all available OpenCL platforms and their devices."""
    platforms = cl.get_platforms()
    for platform in platforms:
        print(
            f"Platform: {platform.name}, Version: {platform.version}, Vendor: {platform.vendor}")
        devices = platform.get_devices()
        for device in devices:
            print(
                f"  Device: {device.name}, Type: {cl.device_type.to_string(device.type)}, Version: {device.version}, Vendor: {device.vendor}")


def print_device_info(device):
    """Prints detailed information about an OpenCL device."""
    print(f"  Device: {device.name}")
    print(f"    Type: {cl.device_type.to_string(device.type)}")
    print(f"    Version: {device.version}")
    print(f"    Vendor: {device.vendor}")
    print(
        f"    Global Memory Size: {device.global_mem_size // (1024 ** 2)} MB")
    print(f"    Local Memory Size: {device.local_mem_size // 1024} KB")
    print(f"    Max Compute Units: {device.max_compute_units}")
    print(f"    Max Work Group Size: {device.max_work_group_size}")
    print(f"    Max Work Item Dimensions: {device.max_work_item_dimensions}")
    print(f"    Max Work Item Sizes: {device.max_work_item_sizes}")


def create_context_and_queue(device_type='GPU', profiling=False, info=False):
    """
    Creates an OpenCL context and command queue for the specified device type.

    Parameters:
    device_type (str): The type of device to create the context for ('GPU' or 'CPU').
    profiling (bool): Whether to enable profiling on the command queue.

    Returns:
    tuple: A tuple containing the OpenCL context and command queue.

    Raises:
    ValueError: If no devices of the specified type are found.
    """
    device_type_map = {
        'GPU': cl.device_type.GPU,
        'CPU': cl.device_type.CPU
    }
    target_device_type = device_type_map.get(device_type, cl.device_type.ALL)

    platforms = cl.get_platforms()
    for platform in platforms:
        devices = platform.get_devices(device_type=target_device_type)
        selected_devices = [
            device for device in devices if device.type & target_device_type]

        if selected_devices:
            ctx = cl.Context(devices=selected_devices)
            queue_properties = cl.command_queue_properties.PROFILING_ENABLE if profiling else 0
            queue = cl.CommandQueue(ctx, properties=queue_properties)
            selected_device = selected_devices[0]
            print(
                f"Using device: {selected_device.name}, Type: {cl.device_type.to_string(selected_device.type)}")
            if(info): print_device_info(selected_device)
            return ctx, queue

        print(f"Platform: {platform.name}, Vendor: {platform.vendor}")
        for device in devices:
            if(info): print_device_info(device)

    raise ValueError(f"No {device_type} devices found")


if __name__ == "__main__":
    # List all available platforms and devices
    list_platforms_and_devices()

    # Create a context and command queue for GPU devices
    try:
        context, queue = create_context_and_queue('GPU', profiling=True)
    except ValueError as e:
        print(e)
        print("Falling back to CPU devices")
        try:
            context, queue = create_context_and_queue('CPU', profiling=True)
        except ValueError as e:
            print(e)
