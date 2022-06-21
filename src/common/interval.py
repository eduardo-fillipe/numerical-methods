class Interval:
    def __init__(self, interval: tuple[float, float]):
        assert interval[0] <= interval[1]
        self._interval: tuple[float, float] = interval
        self._current_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self._current_index += 1
            return self._interval[self._current_index - 1]
        except IndexError:
            self._current_index = 0
            raise StopIteration

    def stepper(self, step: float) -> iter:
        return StepGenerator(self.tuple, step)

    @property
    def tuple(self) -> tuple[float, float]:
        return self._interval

    @property
    def low(self) -> float:
        return self._interval[0]

    @property
    def high(self) -> float:
        return self._interval[1]

    @property
    def size(self) -> float:
        return self.high - self.low

    @property
    def middle_point(self) -> float:
        return (self.low + self.high) / 2

    def __eq__(self, other: 'Interval'):
        if len(self._interval) == len(other._interval):
            if self.low == other.low and self.high == other.high:
                return True
        return False

    def __repr__(self):
        return 'interval' + '[' + str(self.low) + ', ' + str(self.high) + ']'

    def __lt__(self, other):
        return self.size < other.size

    def __gt__(self, other):
        return self.size > other.size

    def __ge__(self, other):
        return self.size >= other.size

    def __le__(self, other):
        return self.size <= other.size

    def __contains__(self, item):
        """
        Checks whether a value belongs to this interval.

        :param item: The value to be evaluated.
        :return: True if the value belongs to the interval, False otherwise
        """
        assert item is not None, 'The value can not be None'

        return self.low <= item <= self.high


class StepGenerator:
    def __init__(self, interval: tuple[float, float], step):
        self.__current_step: float = interval[0]
        self.__interval: tuple[float, float] = interval
        self.__step: float = step

    def __iter__(self):
        return self

    def __next__(self) -> Interval:
        if self.__current_step <= self.__interval[1]:
            result = self.__current_step + self.__step
            c = self.__current_step
            self.__current_step = result
            if result > self.__interval[1]:
                return Interval((c, self.__interval[1]))
            else:
                return Interval((c, result))
        else:
            raise StopIteration


if __name__ == '__main__':
    i1 = Interval((5, 10))
    i2 = Interval((5, 10))

    for i in i2.stepper(0.000000000001):
        print(i)
