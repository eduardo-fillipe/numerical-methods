from enum import Enum, auto
import numpy as np


class GaussPivotingStrategy:
    def __init__(self, solver):
        self.matrix = solver.extended_matrix
        self.solver = solver
        self.logger = solver.iteration_log

    def get_new_pivot_idx(self, relative_point: tuple[int, int], upper=False) -> tuple[int, int]:
        raise NotImplementedError()

    def do_changes(self, old_idx: tuple[int, int], new_idx: tuple[int, int]):
        raise NotImplementedError()

    def restore_non_elementary_operations(self):
        pass


class PartialPivoting(GaussPivotingStrategy):

    def get_new_pivot_idx(self, relative_point: tuple[int, int], upper=False) -> tuple[int, int]:
        k = relative_point[0]
        v = np.array(self.matrix)[k:, k]
        i = abs(v).argmax() + k

        if self.matrix[i, relative_point[1]] == 0:
            return -1, -1
        return i, relative_point[1]

    def do_changes(self, old_idx: tuple[int, int], new_idx: tuple[int, int]):
        DefaultPivoting(self.solver).do_changes(old_idx, new_idx)


class CompletePivoting(GaussPivotingStrategy):

    def __init__(self, solver: 'GaussSolver'):
        super().__init__(solver)
        self.col_changes_stack = []

    def get_new_pivot_idx(self, relative_point: tuple[int, int], upper=False) -> tuple[int, int]:
        k = relative_point[0]
        sub_matrix = abs(self.matrix[k:, k:self.matrix.shape[0]])
        i, j = np.unravel_index(sub_matrix.argmax(), sub_matrix.shape)
        return i + k, j + k

    def do_changes(self, old_idx: tuple[int, int], new_idx: tuple[int, int]):
        change_type = ''
        change_log = ''
        if old_idx[0] != new_idx[0]:
            self.matrix[[old_idx[0], new_idx[0]]] = self.matrix[[new_idx[0], old_idx[0]]]  # change lines
            change_type += 'LINE\n'
            change_log += f'L{old_idx[0]} <-> L{new_idx[0]}\n'

        if old_idx[1] != new_idx[1]:
            self.matrix[:, [old_idx[1], new_idx[1]]] = self.matrix[:, [new_idx[1], old_idx[1]]]  # change columns
            self.col_changes_stack.append((old_idx[1], new_idx[1]))
            change_type += 'COLUMN'
            change_log += f'C{old_idx[1]} <-> C{new_idx[1]}'

        self.logger.log_iteration(self.solver, pivot=old_idx, zeroing_pos=None, factor='', change_line=change_type,
                                  change=change_log, new_matrix=str(self.matrix))

    def restore_non_elementary_operations(self):
        change_log = ''
        for c in range(len(self.col_changes_stack)-1, -1, -1):
            change_log += str(self.col_changes_stack[c]) + '\n'
            self.__undo_change(self.col_changes_stack[c])

        self.logger.log_iteration(self.solver, pivot='', zeroing_pos=None, factor='', change_line='RESTORE',
                                  change=change_log, new_matrix=str(self.matrix))

    def __undo_change(self, change: tuple[int, int]):  # changes the columns of the answer
        self.solver.solution[[change[0], change[1]]] = self.solver.solution[[change[1], change[0]]]


class DefaultPivoting(GaussPivotingStrategy):
    def get_new_pivot_idx(self, relative_point: tuple[int, int], upper=False) -> tuple[int, int]:
        i, j = relative_point

        while i < self.matrix.shape[0]:
            if self.matrix[i, j] != 0:
                return i, j
            i += 1

        return -1, -1

    def do_changes(self, old_idx: tuple[int, int], new_idx: tuple[int, int]):
        if old_idx != new_idx:
            self.matrix[[old_idx[0], new_idx[0]]] = self.matrix[[new_idx[0], old_idx[0]]]
            self.logger.log_iteration(self.solver, pivot=old_idx, zeroing_pos=None, factor='',
                                      change_line='LINE', change=f'L{old_idx[0]} <-> L{new_idx[0]}',
                                      new_matrix=str(self.matrix))


class GaussPivotingStrategyEnum(Enum):
    DEFAULT = auto(),
    PARTIAL = auto(),
    COMPLETE = auto()

    def get_strategy(self, m) -> GaussPivotingStrategy:
        if self.name == self.DEFAULT.name:
            return DefaultPivoting(m)
        if self.name == self.PARTIAL.name:
            return PartialPivoting(m)
        if self.name == self.COMPLETE.name:
            return CompletePivoting(m)
