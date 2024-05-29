# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV


# %%


# Create a DataFrame
def plot_fit_post_xyz(XYZ, x, y):
    # Combine X, Y, Z into a single array for joint KDE
    if isinstance(XYZ, pd.DataFrame):
        XYZ = XYZ.to_numpy()

    X = XYZ[:, 0]
    Y = XYZ[:, 1]
    Z = XYZ[:, 2]
    XYZ = np.vstack([X, Y, Z]).T

    # Fit KDE to the joint distribution
    kde_joint = KernelDensity(bandwidth=0.5, kernel="gaussian")
    kde_joint.fit(XYZ)

    # Define the grid for X and Y
    x_grid = np.linspace(X.min(), X.max(), 50)
    y_grid = np.linspace(Y.min(), Y.max(), 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid_samples = np.vstack([xx.ravel(), yy.ravel()]).T

    # Create a grid of points for Z
    z_grid = np.linspace(Z.min(), Z.max(), 50)

    # Compute the marginal and posterior distributions
    marginal_XY_values = []
    posterior_grid = np.zeros((len(x_grid), len(y_grid), len(z_grid)))

    for i in range(len(x_grid)):
        for j in range(len(y_grid)):
            xy_point = np.array([[x_grid[i], y_grid[j]]])
            xyz_points = np.hstack([np.tile(xy_point, (len(z_grid), 1)), z_grid[:, None]])
            log_density = kde_joint.score_samples(xyz_points)
            density = np.exp(log_density)
            marginal_density = np.trapz(density, z_grid)  # integrate over Z
            marginal_XY_values.append(marginal_density)
            posterior_grid[i, j, :] = density / marginal_density if marginal_density > 0 else np.zeros_like(density)

    marginal_XY = np.array(marginal_XY_values).reshape(xx.shape)

    # Example usage: Get the posterior for a specific value of X and Y
    x_val = 0.5  # Replace with your specific X value
    y_val = 0.5  # Replace with your specific Y value

    # Create the posterior surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Evaluate the posterior on a grid
    posterior_z_mean = np.sum(posterior_grid * z_grid, axis=2) / np.sum(posterior_grid, axis=2)

    # Plot the surface
    ax.plot_surface(xx, yy, posterior_z_mean, cmap="viridis", edgecolor="none", alpha=0.7)

    # Add the initial data points colored by Z
    sc = ax.scatter(X, Y, Z, c=Z, cmap="viridis", s=10, label="Data points")

    # Add color bar
    cbar = fig.colorbar(sc, ax=ax, pad=0.1)
    cbar.set_label("Z values")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Posterior Distribution of Z with Data Points")
    ax.legend()

    def get_posterior(x_val, y_val):
        xy_point = np.array([[x_val, y_val]])
        xyz_points = np.hstack([np.tile(xy_point, (len(z_grid), 1)), z_grid[:, None]])

        # Get the joint density P(X, Y, Z) at these points
        log_joint_density = kde_joint.score_samples(xyz_points)
        joint_density = np.exp(log_joint_density)

        # Get the marginal density P(X, Y) for the given (x_val, y_val)
        x_idx = np.searchsorted(x_grid, x_val)
        y_idx = np.searchsorted(y_grid, y_val)
        marginal_density_xy = marginal_XY[x_idx, y_idx]

        # Compute the posterior
        posterior = joint_density / marginal_density_xy if marginal_density_xy > 0 else np.zeros_like(joint_density)

        return z_grid, posterior

    if x is not None and y is not None:
        return get_posterior(x, y)


# %%
# Generate sample data
np.random.seed(42)
X = np.random.normal(0, 1, 1000)
Y = np.random.normal(0, 1, 1000)
Z = X + Y + np.random.normal(0, 0.5, 1000)  # Z is a function of X and Y with added noise
XYZ = pd.DataFrame(data={"X": X, "Y": Y, "Z": Z})
z, p = plot_fit_post_xyz(XYZ, 0, 0)
# %%
