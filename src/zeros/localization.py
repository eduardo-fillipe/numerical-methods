from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable

from src.common.interval import Interval
from src.common.iteration_loggger import Loggable, IterationLogger

from sympy import N


class LocalizationMethodEnum(Enum):
    DEFAULT = auto


class LocalizationMethodFactory:
    @staticmethod
    def get_method(f: Callable[[float], float],
                   method: LocalizationMethodEnum = LocalizationMethodEnum.DEFAULT,
                   loggers: list[IterationLogger] = None):
        return DefaultLocalizationMethod(f, loggers)


class LocalizationMethod(ABC, Loggable):
    def __init__(self, f: Callable[[float], float], loggers: list[IterationLogger] = None):
        self.f: Callable[[float], float] = f
        self.loggers: list[IterationLogger] = [] if loggers is None else loggers
        super().__init__()

    @abstractmethod
    def locate_zeros(self, search_interval: Interval, step: float) -> list[Interval]:
        pass

    def log_iteration(self, **kwargs):
        for logger in self.loggers:
            logger.log_iteration(self, **kwargs)


class DefaultLocalizationMethod(LocalizationMethod):

    def __init__(self, f: Callable[[float], float], loggers: list[IterationLogger] = None):
        super().__init__(f, loggers)

    def locate_zeros(self, search_interval: Interval, step: float) -> list[Interval]:
        result = []

        current_iteration = 0
        for d in search_interval.stepper(step):
            current_iteration += 1

            f_a = N(self.f(d.low))
            f_b = N(self.f(d.high))

            is_signal_change = DefaultLocalizationMethod.__is_opposite_sign(f_a, f_b)

            if is_signal_change:
                result.append(d)

            self.log_iteration(current_iteration=current_iteration, sub_interval=d, a=d.low, b=d.high, f_a=f_a, f_b=f_b,
                               has_zero=is_signal_change)
        return result

    @staticmethod
    def __is_opposite_sign(n1: float, n2: float) -> bool:
        return (n1 == 0 or n2 == 0) or (n1 < 0 and n2 > 0) or (n1 > 0 and n2 < 0)

    def get_log_columns_names(self) -> list[str]:
        return ['ITERAÇÃO', 'SUB-INTERVALO', 'a', 'b', 'f(a)', 'f(b)', 'Possui zero']

