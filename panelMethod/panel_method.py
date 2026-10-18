import matplotlib.pyplot as plt
import numpy as np
from naca4DigitGenerator import generate_naca4_airfoil


class AirfoilPanels:
    """
    Encapsulates panel geometry and properties for a discretized 2D airfoil.
    """

    def __init__(self, n_panels: int = 100, naca_code: str = '2412', chord: float = 1.0):
        self.n_panels = n_panels
        self.chord = chord

        # 1. Generate node coordinates
        self.x_nd, self.y_nd = generate_naca4_airfoil(
            n_points=n_panels, naca_code=naca_code, chord=self.chord)

        # 2. Panel start and end points
        self.x_start = self.x_nd
        self.y_start = self.y_nd
        self.x_end = np.append(self.x_nd[1:], self.x_nd[0])
        self.y_end = np.append(self.y_nd[1:], self.y_nd[0])

        # 3. Geometric properties
        dx = self.x_end - self.x_start
        dy = self.y_end - self.y_start
        self.l = np.sqrt(dx**2 + dy**2)

        self.sin = dy / self.l
        self.cos = dx / self.l
        self.theta = np.arctan2(dy, dx)

        # 4. Collocation points (midpoints)
        self.x_mid = 0.5 * (self.x_start + self.x_end)
        self.y_mid = 0.5 * (self.y_start + self.y_end)

        # compute beta in local coord
        self.beta = np.zeros((n_panels, n_panels))
        for i in range(n_panels):
            for j in range(n_panels):
                if i == j:
                    self.beta[i, j] = np.pi
                else:
                    dx_ij = self.x_mid[i] - self.x_start[j]
                    dy_ij = self.y_mid[i] - self.y_start[j]
                    x_star = dx_ij * self.cos[j] + dy_ij * self.sin[j]
                    y_star = -dx_ij * self.sin[j] + dy_ij * self.cos[j]

                    phi2 = np.arctan2(y_star, x_star - self.l[j])
                    phi1 = np.arctan2(y_star, x_star)
                    self.beta[i, j] = phi2 - phi1

        self.r = np.zeros((n_panels, n_panels))
        for i in range(n_panels):
            for j in range(n_panels):
                self.r[i, j] = np.sqrt(
                    (self.x_mid[i] - self.x_nd[j])**2 + (self.y_mid[i] - self.y_nd[j])**2)


def solve_HS_panel_method(panels: AirfoilPanels, V_inf: float = 1.0, AoA: float = 0.0):
    """
    Solves the 2D Hess & Smith panel method over a pre-initialized AirfoilPanels object.

    Parameters:
    - panels: AirfoilPanels instance containing geometry, lengths, and angles
    - V_inf: Freestream velocity (default 1.0)
    - AoA: Angle of attack (degrees) of the airfoil (default 0.0)

    Returns:
    - q: numpy 1D array of source distribution on the panels
    - gamma: scalar value of circulation distribution
    """
    alpha = np.deg2rad(AoA)
    n = panels.n_panels

    x_nd, y_nd = panels.x_nd, panels.y_nd
    x_mid, y_mid = panels.x_mid, panels.y_mid
    cos, sin = panels.cos, panels.sin
    theta = panels.theta
    l = panels.l

    # Assemble the matrices
    A = np.zeros((n, n))
    B = np.zeros_like(A)
    C = np.zeros_like(A)
    D = np.zeros_like(A)
    E = np.zeros_like(A)
    F = np.zeros_like(A)
    G = np.zeros_like(A)
    M_top_left = np.zeros_like(A)
    M_top_right = np.zeros((n, 1))

    for i in range(n):
        for j in range(n):
            A[i, j] = - (x_mid[i] - x_nd[j]) * cos[j] - \
                (y_mid[i] - y_nd[j]) * sin[j]

            B[i, j] = (x_mid[i] - x_nd[j])**2 + (y_mid[i] - y_nd[j])**2
            C[i, j] = np.sin(theta[i] - theta[j])
            D[i, j] = np.cos(theta[i] - theta[j])

            E[i, j] = (x_mid[i] - x_nd[j]) * sin[j] - \
                (y_mid[i] - y_nd[j]) * cos[j]

            # Explicitly handle self-induction (i == j)
            if i == j:
                F[i, j] = 0.0
                G[i, j] = -np.pi
            else:
                F[i, j] = np.log(1 + (l[j]**2 + 2.0*A[i, j]*l[j]) / B[i, j])
                G[i, j] = np.arctan2(E[i, j]*l[j], A[i, j]*l[j] + B[i, j])

            M_top_left[i, j] = C[i, j]*F[i, j] / 2 - D[i, j] * G[i, j]
            M_top_right[i, 0] += D[i, j] * F[i, j] / 2 + C[i, j] * G[i, j]

    M_bot_left = np.zeros((1, n))
    M_bot_right = np.zeros((1, 1))
    for j in range(n):
        M_bot_left[0, j] = (D[-1, j] * F[-1, j] / 2 + C[-1, j] * G[-1, j]) + \
                           (D[0, j] * F[0, j] / 2 + C[0, j] * G[0, j])

        M_bot_right -= ((C[-1, j] * F[-1, j] / 2 - D[-1, j] * G[-1, j]) +
                        (C[0, j] * F[0, j] / 2 - D[0, j] * G[0, j]))

    M = 1/(2*np.pi) * np.block([[M_top_left, M_top_right],
                               [M_bot_left, M_bot_right]])

    N = np.zeros(n + 1)
    N[-1] = V_inf * (np.cos(theta[0] - alpha) + np.cos(theta[-1] - alpha))
    N[:-1] = V_inf * np.sin(theta - alpha)

    x_soltn = np.linalg.solve(M, N)
    gamma = x_soltn[-1]
    q = x_soltn[:-1]

    return q, gamma


def compute_cl(panels: AirfoilPanels, gamma: float, V_inf: float):
    cl = 2*gamma/(panels.chord * V_inf) * np.sum(panels.l)
    return cl


def compute_cp(p: AirfoilPanels, gamma: float, q, V_inf: float, AoA: float = 0.0):
    alpha = np.deg2rad(AoA)
    V_t = np.zeros(p.n_panels)
    for i in range(p.n_panels):
        V_t[i] += V_inf * np.cos(p.theta[i] - alpha)
        for j in range(p.n_panels):
            # FIX 4: Minus sign before the source log(...) * cos(theta_i - theta_j) term
            V_t[i] += q[j]/(2*np.pi) *\
                (p.beta[i, j] * np.sin(p.theta[i] - p.theta[j]) -
                 np.log(p.r[i, (j + 1) % p.n_panels]/p.r[i, j]) * np.cos(p.theta[i] - p.theta[j])) +\
                gamma/(2*np.pi) *\
                (np.log(p.r[i, (j + 1) % p.n_panels]/p.r[i, j]) * np.sin(p.theta[i] - p.theta[j]) +
                 p.beta[i, j] * np.cos(p.theta[i] - p.theta[j]))
    cp = np.zeros(p.n_panels)
    cp = 1 - (V_t/V_inf)**2

    return cp


# 1. Instantiate geometry once
airfoil = AirfoilPanels(n_panels=100, naca_code='0018')
V_inf = 1.0
AoA = 0.0

# 2. Pass it to the solver function
q, gamma = solve_HS_panel_method(airfoil, V_inf=V_inf, AoA=AoA)
print("gamma", gamma)

cl = compute_cl(airfoil, gamma, V_inf=V_inf)
print("cl", cl)

cp = compute_cp(airfoil, gamma, q, V_inf, AoA=AoA)

plt.plot(airfoil.x_mid, cp)
plt.xlabel("x/c")
plt.ylabel("Cp")
plt.show()


plt.plot(airfoil.x_nd, airfoil.y_nd)
plt.xlim(-0.5, 1.5)
plt.ylim(-1, 1)
plt.show()
