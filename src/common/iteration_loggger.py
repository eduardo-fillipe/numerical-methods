from abc import ABC, abstractmethod
import pandas as pd
from tabulate import tabulate


class Loggable:

    @abstractmethod
    def get_log_columns_names(self) -> list[str]:
        pass


class IterationLogger(ABC):
    def __init__(self, name=None):
        self.current_iteration = 0
        self.name = name
        self.column_names = []
        self.log_data_list: list[list[int]] = []
        self.log_df: pd.DataFrame = None

    def log_iteration(self, loggable: Loggable, **kwargs):
        if self.is_first_iteration():
            self.column_names = loggable.get_log_columns_names()

        current_iter_params = []
        for arg in kwargs:
            current_iter_params.append(kwargs[arg])

        self.log_data_list.append(current_iter_params)
        self.current_iteration += 1

    def show_log(self):
        self.log_df = pd.DataFrame(self.log_data_list, columns=self.column_names)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        if self.name is not None:
            print(self.name + ':')
        print(tabulate(self.log_df, headers='keys', tablefmt='fancy_grid', showindex=False, disable_numparse=True))

    def is_first_iteration(self):
        return self.current_iteration == 0


class ConsoleLogger(IterationLogger):
    def __init__(self, name=None):
        super().__init__(name)

    def log_iteration_start(self):
        self.current_iteration = 0



