import numpy as np
from random import random
import matplotlib.pyplot as plt

P = np.array([[0.8, 0.2, 0],
            [0.4, 0.3, 0.3],
            [0, 0.7, 0.3]])

T = 100000

N = np.zeros(3, dtype=int)
distr = np.array([1, 0, 0])

state = 0

for i in range(T):
    N[state] += 1
    next_distr_p = random()
    if next_distr_p < P[state][0]:
        state = 0
    elif next_distr_p < P[state][0] + P[state][1]:
        state = 1
    else:
        state = 2

N = N / T

print('N =', N)
print('Sum(N) =', sum(N))
print(np.linalg.matrix_power(P, 300))

A = np.transpose(P)
A[0, 0] -= 1
A[1, 1] -= 1
A[2: ] = 1
b = np.array([0, 0, 1])
print('x =', np.linalg.solve(A, b))
