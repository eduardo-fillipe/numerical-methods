import numpy as np

from src.linear_systems.linear_systems import IterativeLinearSystemSolver


class JacobiSolver(IterativeLinearSystemSolver):

    def __init__(self, extended_matrix: np.ndarray, max_iter=50, desired_error=0.0001):
        super().__init__(extended_matrix, max_iter, desired_error)

    def get_log_columns_names(self) -> list[str]:
        return ['ITERATION', 'F', 'X_k', 'D', 'X_k+1', 'ERRO', 'DESIRED_ERROR']

    def log_iteration(self, error: np.ndarray, x, xi):
        self.iteration_log.log_iteration(self, i=self.curr_iter, f=str(self.F), x=str(x), d=str(self.D), xi=str(xi),
                                         err=str(error), de=str(self.desired_error))

    def __next__(self) -> np.ndarray:
        return (self.F @ self.X) + self.D


class GaussSeidelSolver(IterativeLinearSystemSolver):
    def __init__(self, extended_matrix: np.ndarray, max_iter=50, desired_error=0.0001):
        super().__init__(extended_matrix, max_iter, desired_error)

    def get_log_columns_names(self) -> list[str]:
        return ['ITERATION', 'x_index', 'F', 'X_k', 'D', 'X_k+1', 'ERRO', 'DESIRED_ERROR']

    def log_iteration(self, error: np.ndarray, x, xi):
        self.iteration_log.log_iteration(self, i=self.curr_iter, x_i='-', f=str(self.F), x=str(x), d=str(self.D),
                                         Xi=str(xi), err=str(error), de=str(self.desired_error))

    def __next__(self) -> np.ndarray:
        result = self.X.copy()
        for i in range(self.X.shape[0]):
            x_i = self.F[i] @ result + self.D[i]

            self.iteration_log.log_iteration(self, i=self.curr_iter, x_curr=str(i), f=str(self.F[i]), x=str(result),
                                             d=str(self.D[i]), Xi=str(x_i), err='-', de='-')
            result[i] = x_i

        return result


def main():
    arr = np.array([
        [10, 2, 1, 7],
        [1, 5, 1, -8],
        [2, 3, 10, 6]
    ], dtype=float)

    m1 = np.array([
        [2, 1, 5],
        [1, 4, 6],
    ], dtype=float)

    solver = GaussSeidelSolver(m1, max_iter=6, desired_error=0.05)

    solver.solve()
    print(solver.solution)
    solver.show_iterations()


if __name__ == '__main__':
    main()
