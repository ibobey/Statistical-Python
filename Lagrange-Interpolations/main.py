from numpy import ndarray, arange, array
from numpy import sqrt, sin, power
from lagrange.LagrangeInterpolation import LagrangeInterpolation

x: ndarray = arange(0, 15, 1)
y: ndarray = sin(x) * sqrt(x) / (power(x, 2) + 1)

interp = LagrangeInterpolation(x=x, y=y)
predicted_value: float = interp.predict_value(x=2.25)

predicted_values = interp.predict_values(
    array([1.25, 2.75, 3.63,
           4.47, 5.51, 6.29,
           7.99, 8.001, 9.13])
)

