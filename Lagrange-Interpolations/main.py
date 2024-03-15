from numpy import ndarray, arange, array
from numpy import sqrt, sin, power
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from lagrange.LagrangeInterpolation import LagrangeInterpolation

sns.set_theme(style="white",
              palette="pastel",
              context="notebook")

x: ndarray = arange(0, 16, 0.5)
y: ndarray = sin(x) * sqrt(x) / (power(x, 2) + 1)

interp = LagrangeInterpolation(x=x, y=y)
predicted_value: float = interp.predict_value(x=2.25)

x_values_for_prediction: ndarray = array([1.25,  1.65,
                                          2.15, 2.75, 3.63,
                                          4.47, 5.571, 6.79,
                                          7.799, 8.10071, 9.13,
                                          10.287, 11.31, 12.74])
y_values_predicted: ndarray = interp.predict_values(x_values_for_prediction)

real_vals = sin(x_values_for_prediction) * sqrt(x_values_for_prediction) / (power(x_values_for_prediction, 2) + 1)
square_error = np.abs(np.sum(real_vals ** 2 - y_values_predicted ** 2))

plt.figure(figsize=(9, 5.5))

plt.plot(x, y, "rs-.", label="$f(x)$", alpha=0.85, lw=1)
plt.scatter(x_values_for_prediction, y_values_predicted, s=45, c="#30c730", label="Predicted value")

plt.grid(True, alpha=0.45, ls="--")
plt.legend(loc="best")

plt.title("Lagrange Interpolation Test")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")

plt.text(11, 0.3, f"MSE: {square_error}")
plt.tight_layout()
sns.despine()
plt.show()

print(square_error)
