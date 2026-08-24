import matplotlib.cm as cm
from scipy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp
import time
from Volterra_In_Out_6 import VolterraResponse
from Volterra_Operations import make_symmetrical

# Define coefficients
a1, a2, a3, a4, a5 = 5, -1.5, 0.3, -0.1, 0.05

# Initialize VolterraResponse
V_static = VolterraResponse(kernelName="Volterra_static", use_opencl=True)
V_static.h1[0] = a1
V_static.h2[5, 2] = a2
V_static.h3[1, 6, 2] = a3
V_static.h4[2, 1, 0, 10] = a4
V_static.h5[0, 0, 0, 0, 0] = a5

V_static = make_symmetrical(V_static)

# Plot impulse response of V_static
impulse = np.zeros(18)
impulse[0] = 1
response, _ = V_static.Volterra_In_Out(impulse)

plt.figure(figsize=(8, 6))
plt.plot(response, color='darkblue')
plt.title("Impulse Response of the 5th Order Volterra System")
plt.xlabel("Time [samples]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()


def static_real(x):
    x = np.asarray(x)

    # Create arrays for delayed versions of x
    x_delay1 = np.concatenate(([0], x[:-1]))
    x_delay2 = np.concatenate(([0, 0], x[:-2]))
    x_delay5 = np.concatenate(([0, 0, 0, 0, 0], x[:-5]))
    x_delay6 = np.concatenate(([0, 0, 0, 0, 0, 0], x[:-6]))
    x_delay10 = np.concatenate(([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], x[:-10]))

    # Compute the output using the delayed inputs
    y = (a1 * x +
         a2 * x_delay2 * x_delay5 +
         a3 * x_delay1 * x_delay6 * x_delay2 +
         a4 * x_delay2 * x_delay1 * x * x_delay10 +
         a5 * x * x * x * x * x)

    return y


# Input signal
duration = 20
fs = 4000
t = np.linspace(0, duration, int(fs*duration), endpoint=True)
f0 = 1
f1 = 35
chirp_signal = 3 * chirp(t, f0=f0, f1=f1, t1=duration, method='quadratic')

# Measure calculation time
start_calc_time = time.time()

response_N = static_real(chirp_signal)
response_V, _ = V_static.Volterra_In_Out(chirp_signal)
rmse = np.sqrt(np.mean((response_N - response_V)**2))

calc_time = time.time() - start_calc_time

# Measure plotting time
start_plot_time = time.time()

plt.figure(figsize=(8, 6))
plt.plot(t, chirp_signal, color='blue')
plt.plot(t, response_V, color='green')
plt.plot(t, response_N, color='black', linestyle='--')

# Display RMSE
plt.text(0.1, 0.75, f'RMSE: {rmse:.4f}',
         transform=plt.gca().transAxes, ha='center')
plt.legend(['Input Signal', 'Volterra approx.',
           'System response'], fontsize=12)

plt.title(
    "Comparison of system response and 5th Order Volterra approximation", fontsize=16)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.grid(True)
plt.show()

plot_time = time.time() - start_plot_time

calc_time_minutes = calc_time / 60
plot_time_minutes = plot_time / 60

print(f"Calculation time in minutes: {calc_time_minutes}")
print(f"Plotting time in minutes: {plot_time_minutes}")

# Function to create Volterra response


def create_volterra_response(degree):
    V = VolterraResponse(
        kernelName=f"Volterra_up_to_{degree}d", use_opencl=True)
    if degree >= 1:
        V.h1[0] = a1
    if degree >= 2:
        V.h2[5, 2] = a2
    if degree >= 3:
        V.h3[1, 6, 2] = a3
    if degree >= 4:
        V.h4[2, 1, 0, 10] = a4
    if degree >= 5:
        V.h5[0, 0, 0, 0, 0] = a5
    return V.Volterra_In_Out(chirp_signal)


# Generate responses for each Volterra degree
responses = {}
for degree in range(1, 6):
    response, _ = create_volterra_response(degree)
    responses[degree] = response

# Debugging: Check the shape of responses
for degree in range(1, 6):
    print(f"Degree {degree} response shape: {np.shape(responses[degree])}")

# Plotting all the responses with a color gradient
plt.figure(figsize=(12, 8))
plt.plot(t, chirp_signal, label='Input Signal', color='darkblue', linewidth=2)
plt.plot(t, response_N, label='System Response',
         color='black', linestyle='--', linewidth=2)

# Using 0.1 to 0.9 to avoid very dark or bright colors
colors = cm.inferno(np.linspace(0.1, 0.9, 5))
suffixes = ['1st', '2nd', '3rd', '4th', '5th']

# Plot each response
for degree, color, suffix in zip(range(1, 6), colors, suffixes):
    plt.plot(t, responses[degree],
             label=f'{suffix} order Volterra approx.', color=color, linewidth=2, alpha=0.8)

plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Amplitude", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

############################################################################################################
# single contributions


def create_specific_volterra_response(degree):
    V = VolterraResponse(kernelName=f"Volterra_{degree}d", use_opencl=True)
    if degree == 1:
        V.h1[0] = a1
    elif degree == 2:
        V.h2[5, 2] = a2
    elif degree == 3:
        V.h3[1, 6, 2] = a3
    elif degree == 4:
        V.h4[2, 1, 0, 10] = a4
    elif degree == 5:
        V.h5[0, 0, 0, 0, 0] = a5
    return V.Volterra_In_Out(chirp_signal)


# Calculate the responses for Volterra systems of specific orders
responses = {}
for degree in range(1, 6):
    response, _ = create_specific_volterra_response(degree)
    responses[degree] = response

plt.figure(figsize=(10, 6))
plt.plot(t, chirp_signal, label='Input Signal', color='blue', linewidth=2)
plt.plot(t, response_N, label='System Response',
         color='black', linestyle='--', linewidth=1.8)

colors = cm.inferno(np.linspace(0.1, 0.9, 5))

suffixes = ['1st', '2nd', '3rd', '4th', '5th']
for degree, color, suffix in zip(range(1, 6), colors, suffixes):
    plt.plot(t, responses[degree],
             label=f'{suffix} order contribution', color=color, linewidth=1.8, alpha=0.8)

# plt.title("Contribution of Individual Volterra Orders to the System Response")
plt.xlabel("Time [s]", fontsize=20)
plt.ylabel("Amplitude", fontsize=20)
plt.legend()
plt.grid(True)
plt.show()

############################################################################################################


def create_volterra_response(degree):
    V = VolterraResponse(
        kernelName=f"Volterra_up_to_{degree}d", use_opencl=True)
    if degree >= 1:
        V.h1[0] = a1
    if degree >= 2:
        V.h2[0, 0] = a2
    if degree >= 3:
        V.h3[0, 0, 0] = a3
    if degree >= 4:
        V.h4[0, 0, 0, 0] = a4
    if degree >= 5:
        V.h5[0, 0, 0, 0, 0] = a5
    return V.Volterra_In_Out(chirp_signal)


response_N = static_real(chirp_signal)

# Calculate the responses for Volterra systems including all lower orders up to 5th degree
responses = {}
for degree in range(1, 6):
    response, _ = create_volterra_response(degree)
    responses[degree] = response

# Plotting all the responses
plt.figure(figsize=(12, 8))
plt.plot(t, chirp_signal, label='Input Signal')
plt.plot(t, response_N, label='System Response', color='black', linestyle='--')
for degree, response in responses.items():
    plt.plot(t, response, label=f'Volterra approx. up to {degree}th order')

plt.title("Comparison of system response and Volterra approximations")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

############################################################################################################
# single contributions


def create_specific_volterra_response(degree):
    V = VolterraResponse(kernelName=f"Volterra_{degree}d", use_opencl=True)
    if degree == 1:
        V.h1[0] = a1
    elif degree == 2:
        V.h2[0, 0] = a2
    elif degree == 3:
        V.h3[0, 0, 0] = a3
    elif degree == 4:
        V.h4[0, 0, 0, 0] = a4
    elif degree == 5:
        V.h5[0, 0, 0, 0, 0] = a5
    return V.Volterra_In_Out(chirp_signal)


# Calculate the responses for Volterra systems of specific orders
responses = {}
for degree in range(1, 6):
    response, _ = create_specific_volterra_response(degree)
    responses[degree] = response

# Plotting all the responses
plt.figure(figsize=(12, 8))
plt.plot(t, chirp_signal, label='Input Signal')
plt.plot(t, response_N, label='System Response', color='black', linestyle='--')
for degree, response in responses.items():
    plt.plot(t, response, label=f'Volterra approx. {degree}th Order')

plt.title("Contribution of Individual Volterra Orders to the System Response")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

############################################################################################################
# two tone
duration = 1.0
fs = 4000  # Sampling rate
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
f1 = 10  # First tone frequency
f2 = 50  # Second tone frequency
amplitude1 = 1.0
amplitude2 = 0.7
two_tone_signal = amplitude1 * \
    np.sin(2 * np.pi * f1 * t) + amplitude2 * np.sin(2 * np.pi * f2 * t)

# Pass the signal through the system
response_N = static_real(two_tone_signal)
response_V, _ = V_static.Volterra_In_Out(two_tone_signal)

rmse = np.sqrt(np.mean((response_N - response_V)**2))

# Plot input signal and responses
plt.figure(figsize=(10, 4))
plt.plot(t, two_tone_signal, label='Input Two-Tone Signal')
plt.plot(t, response_N, label='Actual System Output', linestyle='--')
plt.plot(t, response_V, label='Volterra System Output')
plt.text(0.1, 0.75, f'RMSE: {rmse:.4f}',
         transform=plt.gca().transAxes, ha='center')
plt.legend()
plt.title("Input Two-Tone Signal and System Responses")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()

# Function to generate Volterra responses up to a specified degree


def create_volterra_response(degree):
    V = VolterraResponse(
        use_opencl=True, kernelName=f"Volterra_up_to_{degree}d")
    if degree >= 1:
        V.h1[0] = a1
    if degree >= 2:
        V.h2[0, 0] = a2
    if degree >= 3:
        V.h3[0, 0, 0] = a3
    if degree >= 4:
        V.h4[0, 0, 0, 0] = a4
    if degree >= 5:
        V.h5[0, 0, 0, 0, 0] = a5
    return V.Volterra_In_Out(two_tone_signal)


# Generate responses for each Volterra degree
responses = {}
for degree in range(1, 6):
    response, _ = create_volterra_response(degree)
    responses[degree] = response

# Plot responses for different Volterra degrees
plt.figure(figsize=(12, 8))
plt.plot(t, two_tone_signal, label='Input Two-Tone Signal')
plt.plot(t, response_N, label='Actual System Output', linestyle='--')
for degree, response in responses.items():
    plt.plot(t, response, label=f'{degree}th Order Volterra')

plt.title("Comparison of Two-Tone Signal Responses")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)
plt.show()

# Function to compute FFT and return frequency and magnitude


def compute_fft(signal, fs):
    N = len(signal)
    fft_values = fft(signal)
    frequencies = fftfreq(N, 1/fs)
    magnitudes = np.abs(fft_values)
    # Keep only the positive half of the spectrum
    positive_freqs = frequencies[:N // 2]
    positive_magnitudes = magnitudes[:N // 2]
    return positive_freqs, positive_magnitudes


# Compute FFT for the input and responses
input_freqs, input_mags = compute_fft(two_tone_signal, fs)
output_N_freqs, output_N_mags = compute_fft(response_N, fs)
output_V_freqs, output_V_mags = compute_fft(response_V, fs)

# Generate responses for each Volterra degree and compute their FFT
volterra_ffts = {}
for degree in range(1, 6):
    response, _ = create_volterra_response(degree)
    freqs, mags = compute_fft(response, fs)
    volterra_ffts[degree] = (freqs, mags)

# Plot FFT results
plt.figure(figsize=(12, 8))
plt.plot(input_freqs, input_mags, label='Input Two-Tone Signal', linestyle='--')
plt.plot(output_N_freqs, output_N_mags,
         label='Actual System Output', linestyle='-.')

# Add FFTs of Volterra responses
for degree, (freqs, mags) in volterra_ffts.items():
    plt.plot(freqs, mags, label=f'{degree}th Degree Volterra')

plt.xlim(0, 100)
plt.yscale('log')
plt.title("Frequency Spectra of Two-Tone Signal Responses")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude (Log Scale)")
plt.legend()
plt.grid(True)
plt.show()
