import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

slide_params = {
    # Text Sizes (Large for visibility)
    'font.size': 20,           # General default
    'axes.labelsize': 40,      # x and y labels, can be reduced
    'axes.titlesize': 24,      # Title
    'xtick.labelsize': 20,     # Tick numbers
    'ytick.labelsize': 20,
    'legend.fontsize': 18,     # Legend text

    # axes are dark grey => optional
    'axes.edgecolor': "#7D7D7D",    # Dark grey spines
    'xtick.color': '#7D7D7D',       # Dark grey ticks
    'ytick.color': '#7D7D7D',
    'axes.labelcolor': '#7D7D7D',

    # Line & Marker Geometries (Thick for projectors)
    'lines.linewidth': 3,    # Thicker data lines
    'lines.markersize': 2,    # Much larger markers (default was too small)
    'lines.markeredgewidth': 0,  # Remove marker outline for cleaner look

    # Structural Geometries
    'axes.linewidth': 2.0,     # Thicker spines (box)
    'xtick.major.width': 2.0,  # Thicker ticks
    'ytick.major.width': 2.0,
    'xtick.major.size': 8.0,   # Longer ticks
    'ytick.major.size': 8.0,

    # Slide Aesthetics
    'font.family': 'sans-serif',  # Sans-serif is more legible on slides than serif
    'figure.autolayout': True,   # Similar to tight_layout
    'figure.figsize': (12, 9),   # 4:3 Aspect Ratio by default
}
plt.rcParams.update(slide_params)


AoA = "22.0"
Re = "3.0e+05"
ncrits = ["5.0", "7.0", "9.0", "11.0", "13.0"]
files = [
    f"xfoilCpResults/naca0018_re_{Re}_ncrit_{ncrit}_alfa_{AoA}.csv" for ncrit in ncrits]
labels = ncrits
legendTitle = r"$N_{crit}$"

fig, axes = plt.subplots()
colors = cm.coolwarm(np.linspace(0, 1, len(files)))

for i, f in enumerate(files):
    df = pd.read_csv(f)

    axes.plot(df['x'], df['Cp'],
              color=colors[i], label=labels[i])

axes.invert_yaxis()
axes.set_ylabel(r"$C_p$")
axes.set_xlabel(r"$x/c$")
axes.set_title(r"$C_p$ distribution, $\alpha$ = " + f"{AoA} [°]")
axes.grid(True, linestyle='--', alpha=0.6)

fig.legend(labels, loc='upper right', title=legendTitle)

plt.tight_layout()
plt.show()
