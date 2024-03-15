from numpy import ndarray
from numpy import zeros


class LagrangeInterpolation:
    __array_x: ndarray | list
    __array_y: ndarray | list

    __polynom_degree: int
    __array_length: int

    @property
    def polynom_degree(self) -> int:
        """
        Returns the polynom degree
        :return: int
        """
        return self.__polynom_degree

    @polynom_degree.setter
    def polynom_degree(self, value: int) -> None:
        """
        Sets the polynom degree
        :param value:
        :return:
        """
        if not isinstance(value, int) or value <= 1:
            raise Exception("Cannot Supported Polynom Degree")
        self.__polynom_degree = value

    def __init__(self, x: ndarray, y: ndarray) -> None:
        self.__array_x: ndarray = x
        self.__array_y: ndarray = y
        self._setup()

    def _setup(self) -> bool | None:

        if len(self.__array_x) != len(self.__array_y):
            raise Exception("Array must be of equal length")

        if not len(self.__array_x) > 1 or not len(self.__array_y) > 1:
            raise Exception("Array must have more than one element")

        self.__polynom_degree = len(self.__array_x)
        self.__array_length = len(self.__array_x)
        return True

    def predict_value(self, x: float) -> float:
        """
        Predicts the y value of the given x
        :param x: x for the y value to be predicted
        :return: predicted y value by the given x
        """
        summ: float = 0
        for i in range(0, self.__polynom_degree):
            summ += self.__array_y[i] * self.__L_i(x=x, i=i)
        return summ

    def predict_values(self, x: ndarray) -> ndarray:
        summ: ndarray = zeros(len(x), dtype="float64")
        for index in range(0, len(x)):
            for i in range(0, self.__polynom_degree):
                summ[index] += self.__array_y[i] * self.__L_i(x=x[index], i=i)
        return summ

    def __L_i(self, i: int, x: float | ndarray) -> float:
        """
        Product Value of the Lagrange Interpolation Polynomials
        :param i: max iteration number for Product Function
        :param x: The x value of the approximate y value to be calculated
        :return: Product Value
        """
        product: float = 1
        for j in range(0, self.__polynom_degree):
            if j == i:
                continue
            product *= (x - self.__array_x[j]) / (self.__array_x[i] - self.__array_x[j])
        return product
