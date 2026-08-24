from itertools import combinations_with_replacement, permutations
import numpy as np
import matplotlib.pyplot as plt
import Final.Volterra_Operations as vo5
from Volterra_In_Out_6 import VolterraResponse

x = np.ones(12)
g = VolterraResponse(kernelName="g")
k = VolterraResponse(kernelName="k")
t = VolterraResponse(kernelName="t")
# g.h2[3, 5] = 3
g.h2[5, 3] = 3
g.h2[3, 3] = 3
g.h2[1, 4] = 4
g.h2[4, 1] = 4

t = vo5.make_symmetrical(g)

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
    kernel = getattr(t, f"h{i}")
    if np.isscalar(kernel):
        print(f"t{i} is {kernel}")
    else:
        indices = np.transpose(np.nonzero(kernel))
        values = kernel[np.nonzero(kernel)]
        print(f"t{i} indices: {indices}")
        print(f"t{i} values: {values}")

is_symmetrical = vo5.is_symmetrical(g)
print(f"Is g symmetrical? {is_symmetrical}")
is_symmetrical = vo5.is_symmetrical(k)
print(f"Is k symmetrical? {is_symmetrical}")
is_symmetrical = vo5.is_symmetrical(t)
print(f"Is t symmetrical? {is_symmetrical}")

# Number of permutations
indices = [1, 2, 3, 4]
unique_permutations = list(set(permutations(indices)))
print(unique_permutations)
print(len(unique_permutations))
