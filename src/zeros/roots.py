from typing import Callable

from src.common.interval import Interval
from src.common.iteration_loggger import IterationLogger
from src.zeros.localization import LocalizationMethodEnum, LocalizationMethodFactory
from src.zeros.refinement import RefinementMethodEnum, RefinementMethodFactory


class RootCalculator:

    @staticmethod
    def get_roots(f: Callable[[float], float],
                  refinement_method: RefinementMethodEnum,
                  search_interval: tuple[float, float],
                  interval_method: LocalizationMethodEnum = LocalizationMethodEnum.DEFAULT,
                  maximum_iteration: int = 50,
                  interval_search_step: float = 0.01,
                  maximum_allowed_error: float = 0.0000001,
                  refinement_loggers: list[IterationLogger] = None,
                  localization_loggers: list[IterationLogger] = None) -> list[float]:

        localization_method = LocalizationMethodFactory.get_method(f, interval_method, loggers=localization_loggers)
        intervals = localization_method.locate_zeros(Interval(search_interval), interval_search_step)
        refinement = RefinementMethodFactory.get_method(f,
                                                        refinement_method,
                                                        intervals,
                                                        maximum_allowed_error,
                                                        maximum_iteration,
                                                        refinement_loggers)

        return refinement.find_zeros()
