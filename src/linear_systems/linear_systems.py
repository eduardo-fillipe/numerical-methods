import numpy as np
from src.common.iteration_loggger import IterationLogger, Loggable


class SystemWasNotSolvedYet(Exception):
    pass


class UnsolvableSystem(Exception):
    pass


class LinearSystemSolver(Loggable):

    def solve(self) -> np.ndarray:
        raise NotImplemented()

    @property
    def solved(self) -> bool:
        raise NotImplemented()

    @property
    def iteration_log(self) -> IterationLogger:
        raise NotImplemented()

    def show_iterations(self) -> None:
        raise NotImplemented()

    def get_log_columns_names(self) -> list[str]:
        raise NotImplemented()

    @property
    def solution(self) -> np.ndarray:
        raise NotImplemented()


class IterativeLinearSystemSolver(LinearSystemSolver):
    def __init__(self, extended_matrix: np.ndarray, max_iter=50, desired_error=0.0001):
        assert extended_matrix.shape[0] == extended_matrix.shape[1] - 1, \
            "The coefficient matrix needs to be quadratic"
        self._extended_matrix = extended_matrix
        self.max_iter = max_iter
        self.curr_iter = 0
        self.desired_error = desired_error
        self._is_solved = False
        self.order = self._extended_matrix.shape[0]
        self._iter_log = IterationLogger()
        self._solution = np.ndarray(dtype=float, shape=self.order)
        self.F, self.X, self.D = self.__load_components()

    def __load_components(self):
        f, x, d = self.__load_f_component(), self.__load_x_component(), self.__load_d_component()
        return f, x, d

    def __load_f_component(self) -> np.ndarray:
        result = np.ndarray(shape=(self.order, self.order), dtype=float)
        for i in range(self.order):
            for j in range(self.order):
                if j == i:
                    result[i, j] = 0
                else:
                    result[i, j] = -1 * self.extended_matrix[i, j]/self.extended_matrix[i, i]
        return result

    def __load_d_component(self) -> np.ndarray:
        result = np.ndarray(shape=(self.order, 1), dtype=float)
        for i in range(self.order):
            result[i, 0] = self.extended_matrix[i, -1]/self.extended_matrix[i, i]
        return result

    def __load_x_component(self) -> np.ndarray:
        return np.zeros(shape=(self.order, 1), dtype=float)

    def solve(self) -> np.ndarray:
        for i in range(self.max_iter):
            self.curr_iter += 1
            x_i = next(self)
            err = self.get_error(self.X, x_i)
            self.log_iteration(err, self.X, x_i)
            self.X = x_i

            if np.alltrue(err <= self.desired_error):
                break

        self._is_solved = True
        self._solution = self.X.reshape(self.solution.shape)
        return self._solution

    @staticmethod
    def get_error(x: np.ndarray, xi: np.ndarray) -> np.ndarray:
        return np.abs((xi - x)/xi)

    def log_iteration(self, error: np.ndarray, x, xi):
        pass

    def get_log_columns_names(self) -> list[str]:
        raise NotImplemented()

    @property
    def iteration_log(self) -> IterationLogger:
        return self._iter_log

    def show_iterations(self) -> None:
        if not self.solved:
            raise SystemWasNotSolvedYet()
        self.iteration_log.show_log()

    @property
    def solved(self) -> bool:
        return self._is_solved

    @property
    def extended_matrix(self):
        return self._extended_matrix

    @property
    def solution(self) -> np.ndarray:
        return self._solution

    def __next__(self) -> np.ndarray:
        raise NotImplemented()
