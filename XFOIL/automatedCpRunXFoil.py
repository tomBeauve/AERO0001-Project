import pandas as pd
import os
import subprocess


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


def run_single_xfoil(airfoil="0018", viscous=True, re=3.0e5, ncrit=9.0, alpha=5.0, output_name="single_run_result"):
    """
    Automates a single XFOIL evaluation for specified operating parameters.
    """
    # Build command list

    if viscous:
        commands = (
            f"NACA {airfoil}\n"
            f"PANE\n"
            f"OPER\n"
            f"VISC {re}\n"
            f"VPAR\n"
            f"N {ncrit}\n"
            f"\n"  # Exit VPAR submenu back to OPER
            f"ITER 150\n"
            f"ALFA {alpha}\n"
            f"CPWR {output_name}.cp\n"
            f"\n"
            f"QUIT\n"
        )
    else:
        commands = (
            f"NACA {airfoil}\n"
            f"PANE\n"
            f"OPER\n"
            f"ALFA {alpha}\n"
            f"CPWR {output_name}.cp\n"
            f"\n"
            f"QUIT\n"
        )

    execute_xfoil_commands(commands)


def parse_cp_to_csv(cp_filepath, csv_filepath):
    """
    Parses an existing XFOIL .cp surface pressure file and converts it to a CSV file.

    Parameters:
    -----------
    cp_filepath : str
        Path to the input XFOIL .cp file.
    csv_filepath : str
        Path where the output CSV file will be saved.
    """
    if not os.path.exists(cp_filepath):
        print(f"Error: File not found at {cp_filepath}")
        return

    data = []
    with open(cp_filepath, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                # Attempt to parse numerical values, ignoring headers/text
                x_val = float(parts[0])
                y_val = float(parts[1])
                cp_val = float(parts[2])
                data.append({
                    'x': x_val,
                    'y': y_val,
                    'Cp': cp_val
                })
            except ValueError:
                # Skip non-numeric header lines
                continue

    if data:
        df = pd.DataFrame(data)
        df.to_csv(csv_filepath, index=False)
        print(
            f"Successfully converted:\n  Input:  {cp_filepath}\n  Output: {os.path.abspath(csv_filepath)}")
    else:
        print(
            f"Warning: No valid numerical pressure data found in {cp_filepath}")
    if os.path.exists(cp_filepath):
        os.remove(cp_filepath)


if __name__ == "__main__":
    # Define parameters for the single run
    AIRFOIL = "0018"
    IS_VISCOUS = True
    REYNOLDS = 3.0e5
    NCRIT = 5.0
    ALFA = 22.0
    if IS_VISCOUS:
        OUTPUT_PREFIX = f"xfoilCpResults/naca0018_re_{REYNOLDS:.1e}_ncrit_{NCRIT}_alfa_{ALFA}"
    else:
        OUTPUT_PREFIX = f"xfoilCpResults/naca0018_inv_alfa_{ALFA}"

    run_single_xfoil(
        airfoil=AIRFOIL,
        viscous=IS_VISCOUS,
        re=REYNOLDS,
        ncrit=NCRIT,
        alpha=ALFA,
        output_name=OUTPUT_PREFIX
    )

    parse_cp_to_csv(OUTPUT_PREFIX+".cp", OUTPUT_PREFIX + ".csv")
