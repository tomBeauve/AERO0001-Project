import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def generate_re_influence_plots(results_dir="xfoilResults", fixed_ncrit=9.0):
    """
    Generates 4 aerodynamic diagrams showing the influence of Reynolds number 
    at a fixed Ncrit value with a unified common legend.
    """
    pattern = os.path.join(
        results_dir, f"*viscous*_ncrit_{fixed_ncrit:.1f}.csv")
    files = glob.glob(pattern)

    if not files:
        print(
            f"No files found for fixed Ncrit = {fixed_ncrit} in {results_dir}")
        return

    data_list = []
    for f in files:
        df = pd.read_csv(f)
        if not df.empty:
            re_val = df['Reynolds'].iloc[0]
            data_list.append((re_val, df))

    data_list.sort(key=lambda x: x[0])

    # Generate discrete colors along the colormap
    colors = cm.coolwarm(np.linspace(0, 1, len(data_list)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Influence of Reynolds Number (Fixed $N_{{crit}}$ = {fixed_ncrit})", fontsize=16, fontweight='bold')

    for i, (re, df) in enumerate(data_list):
        color = colors[i]
        label = f"Re = {re:.1e}"

        ld_ratio = np.where(df['Cd'] != 0, df['Cl'] / df['Cd'], np.nan)

        axes[0, 0].plot(df['Alpha'], df['Cd'],
                        color=color, lw=1.5, label=label)
        axes[0, 1].plot(df['Alpha'], df['Cl'],
                        color=color, lw=1.5, label=label)
        axes[1, 0].plot(df['Alpha'], ld_ratio,
                        color=color, lw=1.5, label=label)
        axes[1, 1].plot(df['Cd'], df['Cl'], color=color, lw=1.5, label=label)

    _format_subplot(axes[0, 0], r"Angle of Attack $\alpha$ [°]",
                    r"Drag Coefficient $C_d$", r"Drag Polar ($C_d$ vs $\alpha$)")
    _format_subplot(axes[0, 1], r"Angle of Attack $\alpha$ [°]",
                    r"Lift Coefficient $C_l$", r"Lift Curve ($C_l$ vs $\alpha$)")
    _format_subplot(axes[1, 0], r"Angle of Attack $\alpha$ [°]",
                    r"Lift-to-Drag Ratio $C_l / C_d$", r"Efficiency ($C_l / C_d$ vs $\alpha$)")
    _format_subplot(axes[1, 1], r"Drag Coefficient $C_d$",
                    r"Lift Coefficient $C_l$", r"Drag Polar ($C_l$ vs $C_d$)")

    # Unified Common Legend outside the subplots
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(
        0.98, 0.5), title="Reynolds Number", fontsize=11, title_fontsize=12)

    plt.subplots_adjust(right=0.85, top=0.90, bottom=0.08,
                        hspace=0.25, wspace=0.2)
    output_path = os.path.join(
        results_dir, f"influence_re_ncrit_{fixed_ncrit}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Re-influence plot to {output_path}")


def generate_ncrit_influence_plots(results_dir="xfoilResults", fixed_re=3.0e5):
    """
    Generates 4 aerodynamic diagrams showing the influence of Ncrit 
    at a fixed Reynolds number with a unified common legend.
    """
    pattern = os.path.join(
        results_dir, f"*viscous*re_{fixed_re:.1e}_ncrit_*.csv")
    files = glob.glob(pattern)

    if not files:
        print(f"No files found for fixed Re = {fixed_re:.1e} in {results_dir}")
        return

    data_list = []
    for f in files:
        df = pd.read_csv(f)
        if not df.empty:
            ncrit_val = df['Ncrit'].iloc[0]
            data_list.append((ncrit_val, df))

    data_list.sort(key=lambda x: x[0])

    colors = cm.coolwarm(np.linspace(0, 1, len(data_list)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"Influence of $N_{{crit}}$ (Fixed $Re$ = {fixed_re:.1e})", fontsize=16, fontweight='bold')

    for i, (ncrit, df) in enumerate(data_list):
        color = colors[i]
        label = f"$N_{{crit}}$ = {ncrit:.1f}"

        ld_ratio = np.where(df['Cd'] != 0, df['Cl'] / df['Cd'], np.nan)

        axes[0, 0].plot(df['Alpha'], df['Cd'],
                        color=color, lw=1.5, label=label)
        axes[0, 1].plot(df['Alpha'], df['Cl'],
                        color=color, lw=1.5, label=label)
        axes[1, 0].plot(df['Alpha'], ld_ratio,
                        color=color, lw=1.5, label=label)
        axes[1, 1].plot(df['Cd'], df['Cl'], color=color, lw=1.5, label=label)

    _format_subplot(axes[0, 0], r"Angle of Attack $\alpha$ [°]",
                    r"Drag Coefficient $C_d$", r"Drag Polar ($C_d$ vs $\alpha$)")
    _format_subplot(axes[0, 1], r"Angle of Attack $\alpha$ [°]",
                    r"Lift Coefficient $C_l$", r"Lift Curve ($C_l$ vs $\alpha$)")
    _format_subplot(axes[1, 0], r"Angle of Attack $\alpha$ [°]",
                    r"Lift-to-Drag Ratio $C_l / C_d$", r"Efficiency ($C_l / C_d$ vs $\alpha$)")
    _format_subplot(axes[1, 1], r"Drag Coefficient $C_d$",
                    r"Lift Coefficient $C_l$", r"Drag Polar ($C_l$ vs $C_d$)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center right', bbox_to_anchor=(
        0.98, 0.5), title="Amplification Factor", fontsize=11, title_fontsize=12)

    plt.subplots_adjust(right=0.85, top=0.90, bottom=0.08,
                        hspace=0.25, wspace=0.2)
    output_path = os.path.join(
        results_dir, f"influence_ncrit_re_{fixed_re:.1e}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved Ncrit-influence plot to {output_path}")


def _format_subplot(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)


if __name__ == "__main__":
    generate_re_influence_plots(results_dir="xfoilResults", fixed_ncrit=9.0)
    generate_ncrit_influence_plots(results_dir="xfoilResults", fixed_re=3.0e5)
