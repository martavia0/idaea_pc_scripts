import numpy as np

# Define your observed data: y values, independent variable values (x1, x2, x3)
y = np.array([...])  # Array of observed y values
x1 = np.array([...])  # Array of observed x1 values
x2 = np.array([...])  # Array of observed x2 values
x3 = np.array([...])  # Array of observed x3 values

# Build the design matrix X
X = np.vstack((np.ones_like(x1), x1, x2, x3)).T

# Use numpy's linear algebra solver to find the coefficients
coefficients = np.linalg.lstsq(X, y, rcond=None)[0]

# Extract coefficients for each variable
beta_0 = coefficients[0]
beta_1 = coefficients[1]
beta_2 = coefficients[2]
beta_3 = coefficients[3]

# Print the coefficients
print("Intercept (β₀):", beta_0)
print("Coefficient for x₁ (β₁):", beta_1)
print("Coefficient for x₂ (β₂):", beta_2)
print("Coefficient for x₃ (β₃):", beta_3)

# In case we want non-negative solutions for the MLR:
import numpy as np
from scipy.optimize import nnls

# Define your observed data: y values, independent variable values (x1, x2, x3)
y = np.array([...])  # Array of observed y values
x1 = np.array([...])  # Array of observed x1 values
x2 = np.array([...])  # Array of observed x2 values
x3 = np.array([...])  # Array of observed x3 values

# Build the design matrix X
X = np.vstack((np.ones_like(x1), x1, x2, x3)).T

# Use non-negative least squares to find the coefficients
coefficients, _ = nnls(X, y)

# Extract coefficients for each variable
beta_0 = coefficients[0]
beta_1 = coefficients[1]
beta_2 = coefficients[2]
beta_3 = coefficients[3]

# Print the coefficients
print("Intercept (β₀):", beta_0)
print("Coefficient for x₁ (β₁):", beta_1)
print("Coefficient for x₂ (β₂):", beta_2)
print("Coefficient for x₃ (β₃):", beta_3)