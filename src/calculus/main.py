import sympy
from sympy import ln, Symbol
from sympy.utilities import lambdify

from src.calculus.differentiation import Derivative
from src.calculus.integration_method import RectangleMethod, TrapezeMethod, SimpsonMethod
from src.common.interval import Interval
from sympy import cos, ln


def f(x):
    return x * x + 2 + ln(x)


def f20(x):
    return (cos(ln(x)) ** 2)/(x-1)


def f23(x):
    return ((52*(x-2)*(x-3)*(x-6))/36) + ((48*x*(x-3)*(x-6))/8) - ((104*x*(x-2)*(x-6))/9) + ((80*x*(x-2)*(x-3))/72)


def f21(x):
    return (x ** 2)*ln(x + 2)


def p3(x):
    return x*x * cos(x/2)


def main():
    print(f20(-0.000000001))

    method = Derivative(f20, 0, 0.000000001)
    x = Symbol("x")
    d = f20(x).diff()
    g = lambdify(x, d)

    print(method.diff())
    method.show_iterations()

    print(ln(-0.000000001))


def p3_a():
    method = SimpsonMethod(p3, Interval((0, 3)), 6)
    print(method.integrate())
    method.show_iterations()


def p3_b():
    method = Derivative(p3, 1, 0.0001)
    print(method.second_diff())
    method.show_iterations()

    x = Symbol("x")
    d = p3(x).diff().diff()
    g = lambdify(x, d)

    print(str(d) + ": " + str(g(1)))



if __name__ == '__main__':
    p3_b()
