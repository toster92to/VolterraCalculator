''' in this file I am testing the concatenation of two volterra systems of different orders. I am using the VolterraResponse class to create the systems and the Volterra_Operations_5 to concatenate them. I am using a random input signal and I am plotting the output of the 2nd order system, the 3rd order system and the concatenated system. The output of the concatenated system should be the same as the output of the 3rd order system. '''
import numpy as np
import matplotlib.pyplot as plt
import Volterra_Operations as vo5
from Volterra_In_Out_6 import VolterraResponse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

x = np.sin(np.linspace(0, 2*np.pi, 50))+2
# x = np.ones(12)
g = VolterraResponse(kernelName="g", use_opencl=True)
k = VolterraResponse(kernelName="k", use_opencl=True)
linearized = VolterraResponse(kernelName="linearized", use_opencl=True)
system_concatenate_sym = VolterraResponse(
    kernelName="system_concatenate_sym", use_opencl=True)
system_concatenate = VolterraResponse(
    kernelName="system_concatenate", use_opencl=True)

# g.h1[0] = 1
# g.h1[1] = 1
# g.h1[2] = 0.1

# g.h2[0, 0] = 0.03
# g.h2[2, 1] = 0.02
# g.h2[1,3] = 1

# g.h3[0, 0, 0] = 3
# g.h3[0, 0, 1] = 2
# g.h3[2, 0, 0] = 1

# g.h4[0, 0, 0, 0] = 1
# g.h4[2, 0, 0, 1] = 1
# g.h4[3, 0, 0, 2] = 3

# g.h5[0, 0, 0, 0, 0] = 120
# g.h5[3, 0, 0, 0, 1] = 120
# g.h5[0, 1,3, 2, 0] = 120
# g.h5[0, 0, 0, 0, 1] = 120


# k.h1[0] = 1
# k.h1[3] = 3

# k.h2[2, 1] = 120
# k.h2[2, 2] = 0.3
# k.h2[0, 4] = 120

# k.h3[0, 0, 0] = 3
# k.h3[0, 0, 1] = 2
# k.h3[2, 0, 0] = 1

# k.h4[0, 0, 0, 0] = 1
# k.h4[2, 0, 0, 1] = 120
# k.h4[3, 0, 0, 2] = 3

# k.h5[0,0,0,0,0] = 1
# k.h5[0, 0, 0, 0, 1] = 120
# k.h5[3, 0, 0, 0, 1] = 120
# k.h5[0, 0, 0, 2, 0] = 120
# k.h5[3,2,1,4,0]=240


# k.h4[:6, :6, :6, :6] = np.random.rand(6, 6, 6, 6).astype(np.float32)
k.h2[:3, :3] = np.random.rand(3, 3).astype(np.float32)
g.h2[:3, :3] = np.random.rand(3, 3).astype(np.float32)
g.h1[:3] = np.random.rand(3).astype(np.float32)


# parameter test for the 4th degree
# g.h1[0] = 1
# g.h3[0, 0, 0] = 3
# k.h2[0, 0] = 2

k = vo5.make_symmetrical(k)
is_k_symmetrical = vo5.is_symmetrical(k)
g = vo5.make_symmetrical(g)
is_g_symmetrical = vo5.is_symmetrical(g)
print(f"Is g symmetrical? {is_g_symmetrical}")
print(f"Is k symmetrical? {is_k_symmetrical}")

# system_concatenate = vo5.chainVolterraSystems(g, k, skew=0)
start = time.time()
system_concatenate = vo5.make_symmetrical(
    vo5.chainVolterraSystems_opencl(g, k, skew=0))
end = time.time()
print(f"Time to concatenate: {end-start}")
start = time.time()
concatenated = system_concatenate.Volterra_In_Out(x)
end = time.time()

print(f"Time to calculate output: {end-start}")
# z_1 = (x + np.roll(x, 1))**5

is_symmetrical = vo5.is_symmetrical(system_concatenate)
print(f"Is concatenated symmetrical? {is_symmetrical}")

# Calculate the output of the systems
y = g.Volterra_In_Out(x, oversampleFactor=1, neg_len=0)
z = k.Volterra_In_Out(y, oversampleFactor=1, neg_len=0)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(z, label="Soll", color="blue", linestyle='-', marker='x')
plt.plot(concatenated, label="concatenated",
         color="orange", linestyle='-', marker='o')
# plt.plot(z_1, label="analytisch" )
# plt.plot(t.Volterra_In_Out(x), label="hand-set",
#          color="green", linestyle='-', marker='.')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot((z-concatenated)*100/z, label="error % ",
         color="orange", linestyle='-', marker='o')
plt.text(
    0, 100, f"normalized_rmse = {round(np.sqrt(mean_squared_error(z, concatenated))/ (z.max() - z.min()), 3)}")
plt.text(0, 600, f"R² = {round(r2_score(z, concatenated), 3)}")
# plt.plot(z-t.Volterra_In_Out(x), label="hand-set-Error",
#          color="green", linestyle='-', marker='.')


plt.legend()
plt.savefig('concatenated_system_plot.png')
plt.show()

print((z-concatenated)*100/z)
print("Non-zero elements the kernels:")
for i in range(6):
    kernel = getattr(g, f"h{i}")
    if np.isscalar(kernel):
        print(f"g{i} is {kernel}")
    else:
        indices = np.transpose(np.nonzero(kernel))
        values = kernel[np.nonzero(kernel)]
        print(f"g{i} indices: {indices}")
        print(f"g{i} values: {values}")

for i in range(6):
    kernel = getattr(k, f"h{i}")
    if np.isscalar(kernel):
        print(f"k{i} is {kernel}")
    else:
        indices = np.transpose(np.nonzero(kernel))
        values = kernel[np.nonzero(kernel)]
        print(f"k{i} indices: {indices}")
        print(f"k{i} values: {values}")

for i in range(6):
    kernel = getattr(system_concatenate, f"h{i}")
    if np.isscalar(kernel):
        print(f"v{i} is {kernel}")
    else:
        indices = np.transpose(np.nonzero(kernel))
        values = kernel[np.nonzero(kernel)]
        print(f"v{i} indices: {indices}")
        print(f"v{i} values: {values}")

linearized = vo5.make_symmetrical(
    vo5.chainVolterraSystems_opencl(g, vo5.invertKernels(g, use_opencl=True)))

linearized_output = linearized.Volterra_In_Out(
    x, oversampleFactor=1, neg_len=0)

nl_output = g.Volterra_In_Out(x, oversampleFactor=1, neg_len=0)
plt.figure()
plt.plot(nl_output, label="non linear output", marker='o')
plt.plot(linearized_output, label="linearized output", marker='x')
plt.legend()
plt.savefig('linearized_system_plot.png')
plt.show()

for i in range(6):
    kernel = getattr(linearized, f"h{i}")
    if np.isscalar(kernel):
        print(f"v{i} is {kernel}")
    else:
        indices = np.transpose(np.nonzero(kernel))
        values = kernel[np.nonzero(kernel)]
        print(f"l{i} indices: {indices}")
        print(f"l{i} values: {values}")
