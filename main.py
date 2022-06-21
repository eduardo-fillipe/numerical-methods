import sympy

from src.common.interval import Interval
from src.common.iteration_loggger import ConsoleLogger
from src.zeros.localization import DefaultLocalizationMethod
from src.zeros.refinement import NewtonMethod, FalsePositionMethod, BisectionMethod
from sympy import ln, exp, cos, sin

if __name__ == '__main__':

    def f(x):
        return x ** 2 - x * ln(x) - 2

    refinement_logger = ConsoleLogger('Refinamento')
    localization_logger = ConsoleLogger('Localização')

    zero_locator = DefaultLocalizationMethod(f, loggers=[localization_logger])
    # zeros_interval = zero_locator.locate_zeros(Interval((3, 4)), 0.5)

    refiner = FalsePositionMethod(f=f, intervals=[Interval((1.5, 2))], loggers=[refinement_logger], desired_error=0.001,
                                  max_iterations=6)
    print(refiner.find_zeros())

    refinement_logger.show_log()
    localization_logger.show_log()
