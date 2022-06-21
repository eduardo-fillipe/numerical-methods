from typing import Callable

from sympy import Symbol, lambdify

from src.common.iteration_loggger import Loggable, IterationLogger


class DifferentiationMethod(Loggable):
    def diff(self):
        raise NotImplemented()

    def show_iterations(self):
        raise NotImplemented()

    def get_log_columns_names(self) -> list[str]:
        raise NotImplemented()


class Derivative(DifferentiationMethod):
    def __init__(self, f: Callable[[float], float], point: float, h: float, symbol: Symbol = None):
        if symbol is None:
            symbol = Symbol('x')
        self.symbol = symbol
        self.point = point
        self.f = lambdify(self.symbol, f(self.symbol))
        self.h = h
        self.logger = IterationLogger(f"Derivative for: {f(self.symbol)}")

    def diff(self):
        result = (self.f(self.point + self.h) - self.f(self.point - self.h))/(2*self.h)
        self.logger.log_iteration(self, p0=self.f(self.point), p1=self.f(self.point + self.h),
                                  p2=self.f(self.point - self.h), p3=2*self.h, p4=result)
        return result

    def second_diff(self):
        f = self.f
        a = self.point
        h = self.h
        result = (f(a + h) - 2 * f(a) + f(a - h))/(h * h)
        self.logger.log_iteration(self, p0=self.f(self.point), p1=self.f(self.point + self.h),
                                  p2=self.f(self.point - self.h), p3=2*self.h, p4=result)
        return result

    def show_iterations(self):
        return self.logger.show_log()

    def get_log_columns_names(self) -> list[str]:
        return ["f(a)", "f(a + h)", "f(a - h)", "2h", "f'(a)"]
