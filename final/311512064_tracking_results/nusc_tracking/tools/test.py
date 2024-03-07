import numpy as np

dist = np.array([
    [0, 1, 2],
    [3, 6, 5],
    [6, 1, 8],
    [7, 2, 3]
])
x = np.array([])
x = []
x.append(1)
positions1 = np.array([
    [1, 2],
    [0, 2],
    [0, 2]
])
unmatched_positions1_data = [d for d in range(
    positions1.shape[0]) if not (d in positions1[:, 1])]
print(unmatched_positions1_data)
