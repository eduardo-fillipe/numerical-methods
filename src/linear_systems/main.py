import numpy as np

from src.linear_systems.gauss.pivoting_strategy import GaussPivotingStrategyEnum
from src.linear_systems.gauss.solver import GaussSolver
from src.linear_systems.jacobi.solver import GaussSeidelSolver

if __name__ == '__main__':
    matrix1 = [
        [1, 1, 0, 2, 11],
        [1, 1, 2, -1, -3],
        [2, 0, -1, 1, 10],
        [-1, 1, 0, 2, 5]
    ]

    matrix2 = [
        [1, 1, 0, 2, 13],
        [1, 2, -1, 1, 6],
        [2, 0, 1, -1, 7],
        [-1, 1, 1, 1, 8]
    ]

    m3 = [
        [65, 15, 253],
        [15, 5, 52]
    ]

    m4 = [
        [1, 1, 0, 1, 0, 7],
        [-1, -1, 1, 0, 1, 5],
        [1, -1, 0, -1, 1, 0],
        [0, 1, 0, 1, 1, 11],
        [1, 0, 1, 1, 0, 9]
    ]

    m5 = [
        [1, 1, 5],
        [1, -2, -4]
    ]
    m12 = [
        [5, -1, -1, 120],
        [0, -4.6, 1.4, 189],
        [0, 1, -6, 364]
    ]
    m14 = [
        [97, 98, 100],
        [98, 99, 200]
    ]

    m17 = [
        [98, 36, 14, 2865],
        [36, 14, 6, 985],
        [14, 6, 1, 346]
    ]

    m18 = [
        [326.19, 113.12, 39.6, 715.26],
        [113.12, 39.6, 14, 246.21],
        [39.6, 14, 1, 85.53]
    ]
    m19 = [
        [300000, 1000, 13500],
        [1000, 5, 130]
    ]

    p1 = [
        [1, 0, 1, 1, 0, 0, 3],
        [1, -1, 1, 0, 0, 1, 0],
        [1, 1, 0, -1, 1, -1, 1],
        [-1, 1, 1, 0, -1, 1, -5],
        [1, 0, -1, 1, 1, 0, 4],
        [0, 1, -1, -2, 1, 0, 2]
    ]

    p2 = [
        [30, 17.02458, 35],
        [17.02458, 10, 15.78194]
    ]
    np.set_printoptions(suppress=True)
    g = GaussSolver(np.array(p2, dtype=float))
    g.solve()

    g.show_iterations()
    print(g.solution)
