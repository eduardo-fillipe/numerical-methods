import numpy as np

from src.common.iteration_loggger import IterationLogger
from src.common.matrix import is_upper_triangular
from src.linear_systems.gauss.pivoting_strategy import GaussPivotingStrategyEnum
from src.linear_systems.linear_systems import SystemWasNotSolvedYet, UnsolvableSystem, LinearSystemSolver


class GaussSolver (LinearSystemSolver):

    def __init__(self, extended_matrix: np.ndarray,
                 pivoting_strategy_enum: 'GaussPivotingStrategyEnum' = GaussPivotingStrategyEnum.DEFAULT):
        assert extended_matrix.shape[0] == extended_matrix.shape[1] - 1, \
            "The coefficient matrix needs to be quadratic"
        self._is_solved = False
        self._extended_matrix = extended_matrix
        self.order = self._extended_matrix.shape[0]
        self._iter_log = IterationLogger()
        self._solution = np.ndarray(dtype=float, shape=self.order)
        self.pivoting_strategy = pivoting_strategy_enum.get_strategy(self)

    def get_log_columns_names(self) -> list[str]:
        return ['PIVOT', 'ZEROING POS', 'FACTOR', 'CHANGE TYPE', 'CHANGE', 'NEW MATRIX']

    def solve(self) -> np.ndarray:
        if not self.solved:
            self._to_trivial_matrix()
            self._collect_solutions()
            self.pivoting_strategy.restore_non_elementary_operations()
            self._is_solved = True
        return self._solution

    def _to_trivial_matrix(self, logger: IterationLogger = None):
        if is_upper_triangular(self.extended_matrix):
            return
        self._to_upper_triangular_matrix()

    def _to_upper_triangular_matrix(self):
        current_iter = 1
        for i in range(0, self.order - 1):
            pivot = self.__get_pivot(i)
            for j in range(i + 1, self.order):
                self.__update_line(i, j, pivot)
            current_iter += 1

    def _to_lower_triangular_matrix(self):
        current_iter = self._iter_log.current_iteration
        for i in range(1, self.order):
            pivot = self._extended_matrix[i, i]
            for j in range(i - 1, -1, -1):
                self.__update_line(i, j, pivot)
            current_iter += 1

    def __get_pivot(self, line):
        i, j = self.pivoting_strategy.get_new_pivot_idx((line, line))

        if (i == -1 or j == -1) or self.extended_matrix[i][j] == 0:
            self._iter_log.show_log()
            raise UnsolvableSystem(f'Inconsistent pivot {line, line}')

        self.pivoting_strategy.do_changes((line, line), (i, j))

        return self.extended_matrix[line][line]

    def __update_line(self, i, j, pivot):
        current_value = self.extended_matrix[j][i]
        if current_value != 0:
            term = (-1 * current_value) / pivot
            self.extended_matrix[j] += term * self.extended_matrix[i]
            self._iter_log.log_iteration(self, pivot=(i, i), zeroing_pos=(j, i),
                                         factor=term, change_line='', change=None,
                                         new_matrix=str(self.extended_matrix))

    def __get_line_solution(self, i) -> float:
        pivot = self.extended_matrix[i][i]
        if pivot == 0:
            raise UnsolvableSystem(f'The coefficient [{i}][{i}] is zero.')

        def get_sum_of_terms() -> float:
            result = 0
            for j in range(i + 1, self.order):
                result += self.extended_matrix[i][j] * self._solution[j]
            return result

        return (self.extended_matrix[i][-1] + (-1 * get_sum_of_terms()))/pivot

    def _collect_solutions(self):
        for i in range(self.order - 1, -1, -1):
            self._solution[i] = self.__get_line_solution(i)

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


class GaussJordanSolver(GaussSolver):

    def _to_trivial_matrix(self, logger: IterationLogger = None):
        self._to_upper_triangular_matrix()
        self._to_lower_triangular_matrix()

    def _collect_solutions(self):
        for i in range(self.order):
            self._solution[i] = self.extended_matrix[i][-1] / self.extended_matrix[i][i]
