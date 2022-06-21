from typing import Callable

from sympy import Symbol
from sympy.utilities import lambdify

from src.common.interval import Interval
from src.common.iteration_loggger import Loggable, IterationLogger


class IntegrationMethod(Loggable):
    def integrate(self):
        raise NotImplemented()

    def show_iterations(self):
        raise NotImplemented()

    def get_log_columns_names(self) -> list[str]:
        raise NotImplemented()


class RectangleMethod(IntegrationMethod):

    def __init__(self, f: Callable[[float], float], interval: Interval, n: int, symbol: Symbol = None):
        if symbol is None:
            symbol = Symbol('x')
        self.symbol = symbol
        self.interval = interval
        self.f = lambdify(self.symbol, f(self.symbol))
        self.n = n
        self.logger = IterationLogger(f"Integral for: {f(self.symbol)}")

    def get_log_columns_names(self) -> list[str]:
        return ['interval', 'numeric_interval', 'interval area']

    def integrate(self):
        delta = self.interval.size / self.n
        stepper = self.interval.stepper(delta)
        integral = 0
        i = 1
        for x_i in stepper:
            area = self.f(x_i.high) * delta
            integral += area
            self.log_iteration(interval=i, numeric_interval=x_i, area=area)
            i += 1

        return integral

    def log_iteration(self, **kwargs):
        self.logger.log_iteration(self, **kwargs)

    def show_iterations(self):
        self.logger.show_log()


class TrapezeMethod(RectangleMethod):

    def __init__(self, f: Callable[[float], float], interval: Interval, n: int, symbol: Symbol = None):
        super().__init__(f, interval, n, symbol)

    def integrate(self):
        delta = self.interval.size / self.n
        stepper = self.interval.stepper(delta)
        integral = 0
        i = 1
        for x_i in stepper:
            area = (delta/2) * (self.f(x_i.low) + self.f(x_i.high))
            integral += area
            self.log_iteration(interval=i, numeric_interval=x_i, area=area)
            i += 1

        return integral


class SimpsonMethod(RectangleMethod):
    def __init__(self, f: Callable[[float], float], interval: Interval, n: int, symbol: Symbol = None):
        assert n % 2 == 0, 'n needs to be even'
        super().__init__(f, interval, n, symbol)

    def get_log_columns_names(self) -> list[str]:
        return ['interval', 'x0, x1, x2', 'interval area']

    def integrate(self):
        delta = self.interval.size / self.n
        stepper = self.interval.stepper(delta)
        integral = 0
        xs = [e for e in stepper]

        for i in range(1, len(xs), 2):
            area = (delta/3) * (self.f(xs[i - 1].low) + 4 * self.f(xs[i].low) + self.f(xs[i + 1].low))
            integral += area
            self.log_iteration(interval=(i - 1, i + 1), numeric_interval=f'{xs[i - 1].low}, {xs[i].low}, '
                                                                         f'{xs[i + 1].low}', area=area)

        return integral
