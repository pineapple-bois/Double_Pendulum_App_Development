import time
import tracemalloc
import psutil
import os
import csv
import numpy as np
from scipy.integrate import solve_ivp


def analyse_computation_cost(integrator, ode_system, initial_conditions, time_span, csv_filename, **integrator_args):
    """
    Analyze the resource cost and output data of the simulation and write results to a CSV file.

    Parameters:
    - integrator: The integrator function to use (e.g., solve_ivp).
    - ode_system: The system of ODEs to solve.
    - initial_conditions: The initial conditions for the ODE solver.
    - time_span: The time span for the simulation.
    - csv_filename: The name of the CSV file to write the data.
    - **integrator_args: Additional arguments specific to the chosen integrator.
    """

    # Define the directory where the CSV file will be saved
    csv_dir = 'AnalysisData'
    # Ensure the directory exists
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, csv_filename)

    # Start tracking memory usage
    tracemalloc.start()

    # Track CPU usage
    process = psutil.Process()
    cpu_times_before = process.cpu_times()

    # Record the start time
    start_time = time.time()

    # Run the simulation
    sol = integrator(ode_system, time_span, initial_conditions, **integrator_args)

    # Record the end time
    end_time = time.time()

    # Stop memory tracking
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # Get CPU usage after the simulation
    cpu_times_after = process.cpu_times()

    # Extract the atol and rtol values from integrator_args
    atol = integrator_args.get('atol', None)
    rtol = integrator_args.get('rtol', None)

    # Calculate the resource usage
    resource_usage = {
        "total_time_sec": end_time - start_time,
        "cpu_time_user_sec": cpu_times_after.user - cpu_times_before.user,
        "cpu_time_system_sec": cpu_times_after.system - cpu_times_before.system,
        "memory_usage_current_mb": current / (1024 * 1024),
        "memory_usage_peak_mb": peak / (1024 * 1024),
        "num_integration_steps": len(sol.t) if sol.success else None,
        "final_time_value": sol.t[-1] if sol.success else None,
        "atol": atol,
        "rtol": rtol
    }

    # Write the resource usage data to the CSV file
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        fieldnames = resource_usage.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        # Write the header only if the file does not already exist
        if not file_exists:
            writer.writeheader()

        writer.writerow(resource_usage)
