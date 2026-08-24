from itertools import permutations
import numpy as np
from itertools import combinations_with_replacement


def find_combinations(m):
    combinations = []

    def backtrack(curr_combination, remaining_sum):
        if remaining_sum == 0:
            combinations.append(curr_combination)
            return
        if remaining_sum < 0:
            return

        for i in range(1, remaining_sum + 1):
            backtrack(curr_combination + [i], remaining_sum - i)

    backtrack([], m)
    return combinations


m = int(input("Enter the value of m: "))
result = find_combinations(m)
print("All possible combinations:")
for combination in result:
    print(combination)
print(list(combinations_with_replacement(np.arange(7), 4)))
