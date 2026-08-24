from itertools import permutations
# import Volterra_File_Operations_5_5 as vfo5
from itertools import combinations_with_replacement, permutations, product
import numpy as np
from scipy import signal as sg
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement, permutations
from Volterra_In_Out_6 import VolterraResponse
import copy

# input_list = [0,0,0,1,1]

# unique_permutations = list(set(permutations(input_list)))


# unique_permutations_as_lists = [list(perm) for perm in unique_permutations]

# print(len(set(input_list)))

# print(unique_permutations_as_lists)
# print(len(unique_permutations_as_lists))

# h0 = 0
# h1 = np.zeros(10)
# h2 = np.zeros(shape=(6, 6))
# h3 = np.zeros(shape=(6, 6, 6))
# h4 = np.zeros(shape=(6, 6, 6, 6))
# h5 = np.zeros(shape=(6, 6, 6, 6, 6))

# Call the function to write the kernels to CSV files:
# g = VolterraResponse()
# vfo5.writeKernels(g, name='g')
# Call the function to read the kernels from the CSV files:
# a = vfo5.readKernels(name='g')
# # Compare the kernels:
# print(a.h0 == g.h0)
# print(np.all(a.h1 == g.h1))
# print(np.all(a.h2 == g.h2))
# print(np.all(a.h3 == g.h3))
# print(np.all(a.h4 == g.h4))
# print(np.all(a.h5 == g.h5))


# def count_unique_and_permutations(input_list):
#     # Calculate the number of unique elements in the list
#     unique_count = len(set(input_list))

#     # Generate unique permutations
#     unique_permutations = list(set(permutations(input_list)))

#     # Convert permutations from tuples to lists
#     unique_permutations_as_lists = [list(perm) for perm in unique_permutations]

#     # Print the results
#     print("Number of unique elements:", unique_count)
#     print("Unique permutations:", unique_permutations_as_lists)
#     print("Number of unique permutations:", len(unique_permutations_as_lists))

# # Example usage
# input_list = [0, 0, 0, 1, 1]
# count_unique_and_permutations(input_list)

import itertools
import numpy as np


neg_len = 0
len5 = 5

# Generate combinations with replacement
combinations = list(itertools.combinations_with_replacement(
    range(-neg_len, len5 - neg_len), 5))

for tau1, tau2, tau3, tau4, tau5 in combinations:
    unique_indices = [tau1, tau2, tau3, tau4]

    # Calculate the number of permutations
    num_permutations = np.math.factorial(
        4) // np.prod([np.math.factorial(unique_indices.count(tau)) for tau in set(unique_indices)])

    # Print the unique indices and their permutations
    print(
        f"Unique indices: {unique_indices}, Number of permutations: {num_permutations}")

    # Generate and print all the permutations
    permutations = set(itertools.permutations(unique_indices))
    for perm in permutations:
        print(perm)
