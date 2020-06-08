import numpy as np
from random import random
import matplotlib.pyplot as plt

P = np.array([[0.8, 0.15, 0.05],
              [0.3, 0.6, 0.1],
              [0, 0, 1]])

N = 100000


def calc_steps(start_state):
    total_steps = 0
    for n in range(N):
        state = start_state
        steps = 0
        while state != 2:
            steps += 1
            change_prob = random()
            if change_prob < P[state][0]:
                state = 0
            elif change_prob < P[state][0] + P[state][1]:
                state = 1
            else:
                state = 2
        total_steps += steps
    return total_steps / N


avg_steps = []
avg_steps.append(calc_steps(0))
avg_steps.append(calc_steps(1))
print("modeling: " + str(avg_steps))

A = np.array([[1 - P[0, 0], -P[0, 1]],
                   [-P[1, 0], 1 - P[1, 1]]])
A_inv = np.linalg.inv(A)
b = np.array([1, 1])
print("calculated: " + str(np.dot(A_inv, b)))
