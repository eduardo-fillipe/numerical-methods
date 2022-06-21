from abc import ABC, abstractmethod
from typing import Callable, Optional

import sympy

from src.common.interval import Interval
from src.common.iteration_loggger import Loggable, IterationLogger
from enum import Enum, auto

import sympy as sp


class RefinementMethodEnum(Enum):
    BISECTION = auto()
    NEWTON = auto()
    FALSE_POSITION = auto()


class RefinementMethodFactory:

    @staticmethod
    def get_method(f: Callable[[float], float],
                   refinement_method: RefinementMethodEnum,
                   intervals: list[Interval],
                   maximum_allowed_error: float = 0.0001,
                   max_iterations: int = 50,
                   loggers: list[IterationLogger] = None) -> 'RefinementMethod':

        if refinement_method == RefinementMethodEnum.BISECTION:
            return BisectionMethod(f,
                                   intervals,
                                   maximum_allowed_error,
                                   loggers=loggers)

        elif refinement_method == RefinementMethodEnum.NEWTON:
            return NewtonMethod(f,
                                intervals,
                                max_iterations,
                                maximum_allowed_error,
                                loggers=loggers)

        elif refinement_method == RefinementMethodEnum.FALSE_POSITION:
            return FalsePositionMethod(f,
                                       intervals,
                                       max_iterations,
                                       maximum_allowed_error,
                                       loggers=loggers)


class RefinementMethod(ABC, Loggable):
    def __init__(self, f: Callable[[float], float], intervals: list[Interval], error: float, max_iterations: int = 50,
                 loggers=None):
        self.intervals: [Interval] = intervals
        self.desired_error = error
        self.f: Callable[[float], float] = f
        self.loggers: list[IterationLogger] = loggers
        self.max_iterations = max_iterations

    def find_zeros(self) -> list[Optional[float]]:
        result: list[float] = []

        for i, interval in enumerate(self.intervals):
            result.append(self._find_zero(interval, interval_number=i))

        return result

    def log_iteration(self, **kwargs):
        if self.loggers is not None:
            for logger in self.loggers:
                logger.log_iteration(self, **kwargs)

    @abstractmethod
    def get_log_columns_names(self) -> list[str]:
        pass

    @abstractmethod
    def _find_zero(self, interval: Interval, **kwargs) -> Optional[float]:
        """
        This function returns the root of the function inside a given interval. If the root do not exist or the
        method do not converge, returns None.

        :param interval: The interval containing 1 zero to be refined
        :param kwargs: Some additional arguments that the methods may need

        :return: The value of the root inside de interval or None if the method do not converge
        """
        pass


class BisectionMethod(RefinementMethod):
    def __init__(self, f: Callable[[float], float],
                 intervals: list[Interval],
                 desired_error: float = 0.000001,
                 loggers: list[IterationLogger] = None):
        super().__init__(f, intervals, desired_error, loggers=loggers)

    def get_log_columns_names(self) -> list[str]:
        return ['ITERAÇÃO', 'INTERVALO', 'ALFA', 'BETA', 'a', 'b', 'PONTO MÉDIO', 'f(a)', 'f(b)', 'f(ponto médio)',
                'ERRO ATUAL', 'TIPO', 'ERRO DESEJADO']

    def _find_zero(self, interval: Interval, **kwargs) -> float:
        a = interval.low
        b = interval.high
        current_iteration = 0
        while b - a > self.desired_error:
            current_iteration += 1
            log_a = a
            log_b = b

            f_a = self.f(a)
            if f_a == 0:
                self.log_iteration(interation=current_iteration, interval=interval, alpha=interval.low,
                                   beta=interval.high, a=log_a, b=log_b, ponto_medio=None, f_a=f_a, f_b=None,
                                   f_middle=None, error=0, _type='R - f(a) = 0', desired_error=self.desired_error)
                return a

            f_b = self.f(b)
            if f_b == 0:
                self.log_iteration(interation=current_iteration, interval=interval, alpha=interval.low,
                                   beta=interval.high, a=log_a, b=log_b, ponto_medio=None,
                                   f_a=f_a, f_b=f_b, f_middle=None, error=0, _type='R - f(b) = 0',
                                   desired_error=self.desired_error)
                return b

            middle = BisectionMethod.get_middle(a, b)
            f_middle = self.f(middle)
            if f_middle == 0:
                return middle

            has_different_sign_a_middle = BisectionMethod.__has_different_sign(f_a, f_middle)

            if has_different_sign_a_middle:
                b = middle
            else:
                a = middle

            self.log_iteration(interation=current_iteration, interval=interval, alpha=interval.low, beta=interval.high,
                               a=log_a, b=log_b, ponto_medio=middle, f_a=f_a, f_b=f_b, f_middle=f_middle,
                               error=log_b - log_a, _type='I', desired_error=self.desired_error)

        self.log_iteration(interation=current_iteration, interval=interval, alpha=interval.low, beta=interval.high,
                           a=a, b=b, ponto_medio=BisectionMethod.get_middle(a, b), f_a=self.f(a), f_b=self.f(b),
                           f_middle=self.f(BisectionMethod.get_middle(a, b)), error=b - a, _type='R',
                           desired_error=self.desired_error)

        return BisectionMethod.get_middle(a, b)

    @staticmethod
    def __has_different_sign(n1: float, n2: float):
        return n1 < 0 and n2 > 0 or n1 > 0 and n2 < 0

    @staticmethod
    def get_middle(a: float, b: float) -> float:
        return (a + b) / 2


class NewtonMethod(RefinementMethod):
    def __init__(self, f: Callable[[float], float],
                 intervals: list[Interval],
                 max_iterations: int = 50,
                 error: float = 0.000001,
                 initial_values: list[float] = None, loggers=None):
        if initial_values is None:
            initial_values = [None] * len(intervals)

        assert len(intervals) == len(initial_values), 'Each interval must have an initial value.'

        self._symbol = sp.Symbol('x')
        self._symbolic_diff = sp.diff(f(self._symbol))
        self._diff = sp.lambdify(self._symbol, self._symbolic_diff)
        self._initial_values: list[float] = initial_values if initial_values is not None else []

        super().__init__(f, intervals, error, max_iterations, loggers)

    def get_log_columns_names(self) -> list[str]:
        return ['ITERAÇÃO', 'INTERVALO', 'f\'', 'x_n', 'f(x_n)', 'f\'(x_n)', 'Y(x_n)', 'ERRO', 'ERRO DESEJADO']

    def _find_zero(self, interval: Interval, **kwargs) -> Optional[float]:
        """
        This function calculates the root of f in the given interval, using the Newton Methods.

        If the method do not converge (the maximum of iterations is reached), None is returned.

        :param interval: The interval containing 1 zero to be refined
        :param kwargs: Some additional arguments that the methods may need

        :return: The value of the root inside de interval or None if the method do not converge
        """

        initial_value = self._initial_values[kwargs['interval_number']]

        if initial_value is None:
            initial_value = interval.middle_point

        assert initial_value in interval, f'{str(initial_value)} do not belongs to {str(interval)}'

        current_iter: int = 1
        x_curr: float = initial_value
        fx_n: float = self.f(x_curr).evalf()
        fx_n_diff: float = self._symbolic_diff.evalf(subs={self._symbol: x_curr}).evalf()
        x_next: float = self._get_next_x(x_curr, fx_n, fx_n_diff)
        current_error: float = self._get_error(x_next, x_curr)

        while self.max_iterations >= current_iter and current_error > self.desired_error:
            self.log_iteration(current_iter=current_iter, interval=interval, f_diff=self._symbolic_diff, x_curr=x_curr,
                               fx_n=fx_n, fx_n_diff=fx_n_diff, yx_n=str(x_next), current_error=current_error,
                               desired_error=self.desired_error)

            x_curr = x_next
            fx_n = self.f(x_curr).evalf()
            fx_n_diff = self._symbolic_diff.evalf(subs={self._symbol: x_curr}).evalf()
            x_next = self._get_next_x(x_next, fx_n, fx_n_diff)
            current_error = self._get_error(x_next, x_curr)

            current_iter += 1

        if current_iter > self.max_iterations:
            self.log_iteration(current_iter=current_iter, interval=interval, f_diff=self._symbolic_diff, x_curr=None,
                               fx_n=None, fx_n_diff=None, yx_n=None, current_error=None,
                               desired_error=self.desired_error)
            return None

        self.log_iteration(current_iter=current_iter, interval=interval, f_diff=self._symbolic_diff, x_curr=x_curr,
                           fx_n=fx_n, fx_n_diff=fx_n_diff, yx_n=str(x_next), current_error=current_error,
                           desired_error=self.desired_error)
        return x_next

    @staticmethod
    def _get_next_x(x_curr, fx_n, fx_n_diff) -> float:
        return x_curr - (fx_n / fx_n_diff)

    @staticmethod
    def _get_error(x_curr: float, x_last: float) -> float:
        return abs((x_curr - x_last)/x_curr)


class FalsePositionMethod(RefinementMethod):

    def __init__(self, f: Callable[[float], float],
                 intervals: list[Interval],
                 max_iterations=50,
                 desired_error: float = 0.000001,
                 loggers: list[IterationLogger] = None):
        super().__init__(f, intervals, desired_error, max_iterations=max_iterations, loggers=loggers)

    def get_log_columns_names(self) -> list[str]:
        return ['ITERAÇÃO', 'INTERVALO', 'ALFA', 'BETA', 'xi', 'f(alfa)', 'f(beta)', 'f(xi)',
                'ERRO ATUAL', 'ERRO DESEJADO']

    def _find_zero(self, interval: Interval, **kwargs) -> Optional[float]:
        i = 0
        err = self.desired_error + 1
        alpha = interval.low
        beta = interval.high
        xi = alpha
        x_last = alpha

        while i <= self.max_iterations and err > self.desired_error:
            i += 1
            f_alpha = self.f(alpha).evalf()
            f_beta = self.f(beta).evalf()
            xi = (alpha * f_beta - beta * f_alpha)/(f_beta - f_alpha)
            f_xi = self.f(xi).evalf()
            err = abs(xi - x_last) / abs(xi)

            self.log_iteration(i=i, interval=interval, alpha=alpha, beta=beta, x_i=xi, f_a=f_alpha, f_b=f_beta,
                               f_xi=f_xi, err=err, desired_error=self.desired_error)

            if self.f(alpha).evalf() * f_xi < 0:
                beta = xi
            else:
                alpha = xi

            x_last = xi
            print(xi)

        return xi
