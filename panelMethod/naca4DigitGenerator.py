import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


import numpy as np


def generate_naca4_airfoil(n_points: int = 100, naca_code: str = '0018', chord: float = 1.0):
    """
    Generates clockwise (x, y) coordinates for a NACA 4-digit series airfoil
    with an exact total number of points equal to n_points.

    Parameters:
    - n_points: Total number of nodes around the airfoil loop
    - naca_code: 4-digit string specifying the airfoil (default '0018')
    - chord: Chord length (default 1.0)

    Returns:
    - x_nd, y_nd: 1D numpy arrays of node coordinates forming a clockwise closed loop.
    """
    # Parse NACA 4-digit code parameters
    m = int(naca_code[0]) / 100.0  # Maximum camber
    p = int(naca_code[1]) / 10.0   # Position of maximum camber
    t = int(naca_code[2:]) / 100.0  # Maximum thickness ratio

    # Split total points between lower and upper surfaces
    n_lower = n_points // 2
    n_upper = n_points - n_lower

    beta_lower = np.linspace(np.pi, 0, n_lower + 1)
    x_l = (chord / 2.0) * (1.0 - np.cos(beta_lower))

    beta_upper = np.linspace(0, np.pi, n_upper + 1)
    x_u = (chord / 2.0) * (1.0 - np.cos(beta_upper))

    def get_airfoil_geometry(x_array):
        xc = x_array / chord

        # Thickness distribution T(x)[cite: 1]
        thickness = 10.0 * t * chord * (
            0.2969 * np.sqrt(xc)
            - 0.1260 * xc
            - 0.3537 * (xc**2)
            + 0.2843 * (xc**3)
            - 0.1015 * (xc**4)
        )

        # Camber line Y_bar(x) and its derivative[cite: 1]
        y_cb = np.zeros_like(x_array)
        dy_cb = np.zeros_like(x_array)

        if m > 0 and p > 0:
            idx_front = xc <= p
            idx_rear = xc > p

            y_cb[idx_front] = (m * (chord * xc[idx_front]) /
                               (p**2)) * (2.0 * p - xc[idx_front])
            dy_cb[idx_front] = (2.0 * m / (p**2)) * (p - xc[idx_front])

            y_cb[idx_rear] = (m * chord / ((1.0 - p)**2)) * \
                ((1.0 - 2.0 * p) + 2.0 * p * xc[idx_rear] - xc[idx_rear]**2)
            dy_cb[idx_rear] = (2.0 * m / ((1.0 - p)**2)) * (p - xc[idx_rear])

        theta = np.arctan(dy_cb)
        return thickness, y_cb, theta

    # Evaluate geometry
    thick_l, ycb_l, theta_l = get_airfoil_geometry(x_l)
    thick_u, ycb_u, theta_u = get_airfoil_geometry(x_u)

    # Lower surface coordinates (TE -> LE)[cite: 1]
    x_lower = x_l + 0.5 * thick_l * np.sin(theta_l)
    y_lower = ycb_l - 0.5 * thick_l * np.cos(theta_l)

    # Upper surface coordinates (LE -> TE)[cite: 1]
    x_upper = x_u - 0.5 * thick_u * np.sin(theta_u)
    y_upper = ycb_u + 0.5 * thick_u * np.cos(theta_u)

    # Assemble into a clockwise continuous loop: TE -> Lower -> LE -> Upper -> TE (excluded)
    x_nd = np.concatenate([x_lower, x_upper[1:-1]])
    y_nd = np.concatenate([y_lower, y_upper[1:-1]])
    return x_nd, y_nd
