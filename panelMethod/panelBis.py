from naca4DigitGenerator import generate_naca4_airfoil
import matplotlib.pyplot as plt
import numpy as np


class AirfoilPanels:
    def __init__(self, n_panels: int = 100, naca_code: str = '0018', chord: float = 1.0):
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

        # 5. Precompute local geometric matrices (Slide 44)
        n = self.n_panels
        self.C = np.zeros((n, n))
        self.D = np.zeros((n, n))
        self.F = np.zeros((n, n))
        self.G = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                self.C[i, j] = np.sin(self.theta[i] - self.theta[j])
                self.D[i, j] = np.cos(self.theta[i] - self.theta[j])
                if i == j:
                    self.F[i, j] = 0.0
                    self.G[i, j] = -np.pi
                else:
                    A_ij = -(self.x_mid[i] - self.x_nd[j]) * self.cos[j] - \
                        (self.y_mid[i] - self.y_nd[j]) * self.sin[j]
                    B_ij = (self.x_mid[i] - self.x_nd[j])**2 + \
                        (self.y_mid[i] - self.y_nd[j])**2
                    E_ij = (self.x_mid[i] - self.x_nd[j]) * self.sin[j] - \
                        (self.y_mid[i] - self.y_nd[j]) * self.cos[j]
                    self.F[i, j] = np.log(
                        1.0 + (self.l[j]**2 + 2.0 * A_ij * self.l[j]) / B_ij)
                    self.G[i, j] = np.arctan2(
                        E_ij * self.l[j], A_ij * self.l[j] + B_ij)

        # Consistency with Slide 14/29 notation: beta_ij = -G_ij, ln(r_{i,j+1}/r_{i,j}) = 0.5 * F_ij
        self.beta = -self.G


def solve_HS_panel_method(panels: AirfoilPanels, V_inf: float = 1.0, AoA: float = 0.0):
    alpha = np.deg2rad(AoA)
    n = panels.n_panels
    C, D, F, G = panels.C, panels.D, panels.F, panels.G
    theta = panels.theta

    M_top_left = 0.5 * C * F - D * G
    M_top_right = np.sum(0.5 * D * F + C * G, axis=1, keepdims=True)

    M_bot_left = (0.5 * D[-1, :] * F[-1, :] + C[-1, :] * G[-1, :]) + \
                 (0.5 * D[0, :] * F[0, :] + C[0, :] * G[0, :])
    M_bot_left = M_bot_left.reshape(1, n)

    M_bot_right = -np.sum(M_top_left[-1, :] + M_top_left[0, :]).reshape(1, 1)

    M = (1.0 / (2.0 * np.pi)) * np.block([[M_top_left, M_top_right],
                                          [M_bot_left, M_bot_right]])

    N = np.zeros(n + 1)
    N[:-1] = V_inf * np.sin(theta - alpha)
    N[-1] = V_inf * (np.cos(theta[0] - alpha) + np.cos(theta[-1] - alpha))

    x_soltn = np.linalg.solve(M, N)
    q = x_soltn[:-1]
    gamma = x_soltn[-1]

    return q, gamma


def compute_cp(p: AirfoilPanels, gamma: float, q: np.ndarray, V_inf: float = 1.0, AoA: float = 0.0):
    alpha = np.deg2rad(AoA)
    C, D, F, G = p.C, p.D, p.F, p.G

    # Tangential velocity on each panel (corrected minus sign on source log term)
    V_t = (V_inf * np.cos(p.theta - alpha)
           - (0.5 * D * F + C * G) @ q / (2.0 * np.pi)
           + gamma / (2.0 * np.pi) * np.sum(0.5 * C * F - D * G, axis=1))

    cp = 1.0 - (V_t / V_inf)**2
    return cp


def compute_cl(panels: AirfoilPanels, gamma: float, V_inf: float):
    cl = 2*gamma/(panels.chord * V_inf) * np.sum(panels.l)
    return cl


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
plt.show()
