import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# 1. Set material parameters and compute Young's modulus
# -------------------------------
nu = 0.3               # Poisson's ratio
epsilon_yy = 0.1       # Prescribed strain in y-direction
epsilon_xx = -nu * epsilon_yy  # x-direction strain (-0.03)
epsilon_xy = 0.0       # Shear strain

# Given Lamé parameters (example values, adjust as needed)
lamda = 100.0          # First Lamé parameter (λ)
mu = 80.0              # Second Lamé parameter (shear modulus, μ)

# Compute Young's modulus E using the relation:
E = mu * (3 * lamda + 2 * mu) / (lamda + mu)
print("Young's modulus E =", E)

# -------------------------------
# 2. Compute stress under plane stress conditions
# -------------------------------
# Under plane stress, we typically have sigma_xx = 0 and:
sigma_xx = 0.0
sigma_yy = E / (1 - nu**2) * epsilon_yy
sigma_xy = 0.0

# -------------------------------
# 3. Generate a grid over the (0,1)x(0,1) square domain
# -------------------------------
nx, ny = 32, 32  # Number of grid points in x and y (adjust as needed)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# -------------------------------
# 4. Construct the displacement field (uniform strain solution)
# -------------------------------
# u_x = epsilon_xx * x, and u_y = epsilon_yy * y
u_x = epsilon_xx * X  # u_x = -0.03 * x
u_y = epsilon_yy * Y  # u_y = 0.1 * y

# -------------------------------
# 5. Construct the uniform strain field (constant over the domain)
# -------------------------------
epsilon_xx_field = np.full_like(X, epsilon_xx)
epsilon_yy_field = np.full_like(X, epsilon_yy)
epsilon_xy_field = np.zeros_like(X)

# -------------------------------
# 6. Construct the uniform stress field (under plane stress)
# -------------------------------
sigma_xx_field = np.full_like(X, sigma_xx)
sigma_yy_field = np.full_like(X, sigma_yy)
sigma_xy_field = np.full_like(X, sigma_xy)

# -------------------------------
# 7. Visualize the displacement field (vector plot)
# -------------------------------
plt.figure(figsize=(6,6))
plt.quiver(X, Y, u_x, u_y, color='b')
plt.title("Displacement Field")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True)
plt.savefig("train.png")

# -------------------------------
# 8. Organize and save the training data
# -------------------------------
data = {
    'X': X,
    'Y': Y,
    'u_x': u_x,
    'u_y': u_y,
    'epsilon_xx': epsilon_xx_field,
    'epsilon_yy': epsilon_yy_field,
    'epsilon_xy': epsilon_xy_field,
    'sigma_xx': sigma_xx_field,
    'sigma_yy': sigma_yy_field,
    'sigma_xy': sigma_xy_field
}

np.savez("elastic_training_data.npz", **data)
print("Training data saved to elastic_training_data.npz")
