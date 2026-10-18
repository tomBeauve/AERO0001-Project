import os
import subprocess
import pandas as pd


def run_xfoil_parametric(airfoil_name, re_list, ncrit_list, alpha_start, alpha_end, alpha_step):
    """
    Automates XFOIL execution for inviscid polars and a multi-dimensional 
    matrix of viscous runs varying Re, Ncrit, and Alpha via ASeq.
    """
    # 1. Inviscid Run (Independent of Re and Ncrit)
    print(f"Running Inviscid Analysis for NACA {airfoil_name}...")
    inv_polar_file = f"xfoilResults/naca{airfoil_name}_inv.polar"
    if os.path.exists(inv_polar_file):
        os.remove(inv_polar_file)

    inv_commands = (
        f"NACA {airfoil_name}\n"
        f"PANE\n"
        f"OPER\n"
        f"PACC\n"
        f"{inv_polar_file}\n"
        f"\n"
        f"ITER 150\n"
        f"ASeq {alpha_start} {alpha_end} {alpha_step}\n"
        f"PACC\n"
        f"QUIT\n"
    )
    execute_xfoil_commands(inv_commands)
    parse_polar_to_csv(
        inv_polar_file, f"xfoilResults/naca{airfoil_name}_inviscid.csv", re_val=0.0, ncrit_val=0.0)

    # 2. Viscous Parametric Matrix Runs (Re x Ncrit)
    for re in re_list:
        for ncrit in ncrit_list:
            print(
                f"Running Viscous Analysis: Re = {re:.1e}, Ncrit = {ncrit}...")
            visc_polar_file = f"xfoilResults/naca{airfoil_name}_re_{re:.1e}_ncrit_{ncrit:.1f}.polar"
            if os.path.exists(visc_polar_file):
                os.remove(visc_polar_file)

            # Command sequence including VPAR menu for Ncrit modification
            visc_commands = (
                f"NACA {airfoil_name}\n"
                f"PANE\n"
                f"OPER\n"
                f"VISC {re}\n"
                f"VPAR\n"
                f"N {ncrit}\n"
                f"\n"  # Exit VPAR submenu back to OPER
                f"PACC\n"
                f"{visc_polar_file}\n"
                f"\n"
                f"ITER 250\n"
                f"ASeq {alpha_start} {alpha_end} {alpha_step}\n"
                f"PACC\n"
                f"QUIT\n"
            )

            execute_xfoil_commands(visc_commands)
            parse_polar_to_csv(
                visc_polar_file,
                f"xfoilResults/naca{airfoil_name}_viscous_re_{re:.1e}_ncrit_{ncrit:.1f}.csv",
                re_val=re,
                ncrit_val=ncrit
            )


def execute_xfoil_commands(commands):
    """Sends command string to xfoil.exe via stdin."""
    process = subprocess.Popen(
        ['xfoil.exe'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        _, _ = process.communicate(input=commands, timeout=90)
    except subprocess.TimeoutExpired:
        process.kill()
        print("XFOIL process timed out.")


def parse_polar_to_csv(polar_path, csv_path, re_val, ncrit_val):
    """Parses XFOIL raw polar text output into a structured CSV file with metadata columns."""
    if not os.path.exists(polar_path):
        print(f"Polar file {polar_path} not found due to convergence failure.")
        return

    data = []
    with open(polar_path, 'r') as f:
        lines = f.readlines()

    start_parsing = False
    for line in lines:
        if "----" in line:
            start_parsing = True
            continue
        if start_parsing:
            parts = line.strip().split()
            if len(parts) >= 6:
                try:
                    data.append({
                        'Reynolds': re_val,
                        'Ncrit': ncrit_val,
                        'Alpha': float(parts[0]),
                        'Cl': float(parts[1]),
                        'Cd': float(parts[2]),
                        'Cdp': float(parts[3]),
                        'Cm': float(parts[4]),
                        'Cp_min': float(parts[5])
                    })
                except ValueError:
                    continue

    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")
    else:
        print(f"No valid convergence data found in {polar_path}")


if __name__ == "__main__":
    AIRFOIL = "0018"
    RE_VALUES = [1.5e5, 3.0e5, 4.5e5, 6.0e5]
    # Example set of critical amplification ratios
    NCRIT_VALUES = [5.0, 7.0, 9.0, 11.0, 13.0]
    ALPHA_START = 0.0
    ALPHA_END = 25.0
    ALPHA_STEP = 0.25

    run_xfoil_parametric(AIRFOIL, RE_VALUES, NCRIT_VALUES,
                         ALPHA_START, ALPHA_END, ALPHA_STEP)
