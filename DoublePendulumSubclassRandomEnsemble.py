import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
import csv
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from joblib import Parallel, delayed
from MathFunctions import *


def hamiltonian_first_order_system(model='simple'):
    Heq1, Heq2, Heq3, Heq4 = hamiltonian_system(model)

    LHS_FIRST = sp.Matrix([[Heq1.lhs], [Heq2.lhs], [Heq3.lhs], [Heq4.lhs]])
    RHS_FIRST = sp.Matrix([[Heq1.rhs], [Heq2.rhs], [Heq3.rhs], [Heq4.rhs]])

    MAT_EQ = sp.Eq(LHS_FIRST, RHS_FIRST)

    return MAT_EQ, Heq1.rhs, Heq2.rhs, Heq3.rhs, Heq4.rhs


class DoublePendulum:
    # Class variable for caching
    _cache = {}

    # Declare variables & constants
    t = sp.Symbol("t")
    l1, l2, m1, m2, M1, M2, g = sp.symbols('l1 l2 m1 m2 M1 M2 g', real=True, positive=True)

    # Declare functions
    theta1 = sp.Function('theta1')(t)
    theta2 = sp.Function('theta2')(t)
    p_theta_1 = sp.Function('p_theta_1')(t)
    p_theta_2 = sp.Function('p_theta_2')(t)

    @classmethod
    def _compute_and_cache_equations(cls, model):
        if model not in cls._cache:
            cls._cache[model] = hamiltonian_first_order_system(model)
        return cls._cache[model]

    def __init__(self, parameters, initial_conditions, time_vector,
                 model='simple', integrator=solve_ivp, **integrator_args):
        self.initial_conditions = np.deg2rad(initial_conditions)
        self.time = np.linspace(time_vector[0], time_vector[1], time_vector[2])
        self.parameters = parameters
        self.model = model

        # Get equations for the specified model
        MAT_EQ, eqn1, eqn2, eqn3, eqn4 = self._compute_and_cache_equations(model)
        self.matrix = MAT_EQ

        # Substitute parameters into the equations
        eq1_subst = eqn1.subs(parameters)
        eq2_subst = eqn2.subs(parameters)
        eq3_subst = eqn3.subs(parameters)
        eq4_subst = eqn4.subs(parameters)

        # Lambdify the equations after substitution
        self.eqn1_func = sp.lambdify((theta1, theta2, p_theta_1, p_theta_2, t), eq1_subst, 'numpy')
        self.eqn2_func = sp.lambdify((theta1, theta2, p_theta_1, p_theta_2, t), eq2_subst, 'numpy')
        self.eqn3_func = sp.lambdify((theta1, theta2, p_theta_1, p_theta_2, t), eq3_subst, 'numpy')
        self.eqn4_func = sp.lambdify((theta1, theta2, p_theta_1, p_theta_2, t), eq4_subst, 'numpy')

        # Run the solver
        self.sol = self._solve_ode(integrator, **integrator_args)

    def _system(self, y, t):
        th1, th2, p_th1, p_th2 = y
        system = [
            self.eqn1_func(th1, th2, p_th1, p_th2, t),
            self.eqn2_func(th1, th2, p_th1, p_th2, t),
            self.eqn3_func(th1, th2, p_th1, p_th2, t),
            self.eqn4_func(th1, th2, p_th1, p_th2, t)
        ]
        return system

    def _solve_ode(self, integrator, initial_conditions=None, **integrator_args):
        """
        Solve the system of ODEs using the specified integrator.

        Parameters:
        - integrator: The integrator function to use. Default is scipy's solve_ivp.
        - initial_conditions: The initial conditions for the ODE solver.
        - **integrator_args: Additional arguments specific to the chosen integrator.
        """
        if initial_conditions is None:
            initial_conditions = self.initial_conditions

        # Open the CSV file in append mode
        with open("raw_termination_data.csv", 'a', newline='') as csvfile:
            fieldnames = ['initial_conditions', 'termination_time', 'termination_reason']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Write header if the file is empty
            if csvfile.tell() == 0:
                writer.writeheader()

            try:
                if integrator == odeint:
                    sol = odeint(self._system, initial_conditions, self.time, **integrator_args)
                    return sol

                if integrator == solve_ivp:
                    t_span = (self.time[0], self.time[-1])

                    # Define an event to detect large deviations
                    def event_large_deviation(t, y):
                        max_angle_limit = 10 * np.pi
                        return max_angle_limit - max(np.abs(y[0]), np.abs(y[1]))

                    event_large_deviation.terminal = True  # Stop the integration
                    event_large_deviation.direction = -1  # Trigger when the condition is crossed

                    sol = solve_ivp(lambda t, y: self._system(y, t), t_span, initial_conditions,
                                    t_eval=self.time, events=event_large_deviation, **integrator_args)

                    # Check for early termination due to large deviations
                    if sol.status == 1:
                        termination_time = sol.t_events[0][0]
                        writer.writerow({'initial_conditions': initial_conditions,
                                         'termination_time': termination_time,
                                         'termination_reason': 'Large deviation'})
                        return None

                    # Check for NaNs or Infs
                    if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                        writer.writerow({'initial_conditions': initial_conditions,
                                         'termination_time': sol.t[-1],
                                         'termination_reason': 'NaN/Inf detected'})
                        return None

                    return sol.y.T  # Transpose

                else:
                    raise ValueError("Unsupported integrator")

            except Exception as e:
                print(f"Error encountered during simulation for initial conditions {initial_conditions}: {e}")
                writer.writerow({'initial_conditions': initial_conditions,
                                 'termination_time': 'N/A',
                                 'termination_reason': f'Exception: {str(e)}'})
                return None

    def _calculate_positions(self):
        # Unpack solution for theta1 and theta2
        theta_1, theta_2 = self.sol[:, 0], self.sol[:, 1]

        # Evaluate lengths of the pendulum arms using the provided parameter values
        l_1 = float(self.parameters[l1])
        l_2 = float(self.parameters[l2])

        # Calculate the (x, y) positions of the first pendulum bob
        x_1 = l_1 * np.sin(theta_1)
        y_1 = -l_1 * np.cos(theta_1)

        # Calculate the (x, y) positions of the second pendulum bob
        x_2 = x_1 + l_2 * np.sin(theta_2)
        y_2 = y_1 - l_2 * np.cos(theta_2)

        return x_1, y_1, x_2, y_2

    def time_graph(self):
        plt.style.use('default')  # Reset to the default style
        fig, ax = plt.subplots()
        # Plot settings to match the animation's appearance
        ax.plot(self.time, np.rad2deg(self.sol[:, 0]), color='darkorange', label="θ1", linewidth=2)
        ax.plot(self.time, np.rad2deg(self.sol[:, 1]), color='green', label="θ2", linewidth=2)

        # Set the labels, title, and grid
        ax.set_xlabel('Time / seconds')
        ax.set_ylabel('Angular displacement / degrees')
        ax.set_title('Time Graph', fontname='Courier New', fontsize=16)

        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.legend(loc='best')
        return fig

    def phase_path(self):
        plt.style.use('default')  # Reset to the default style
        fig, ax = plt.subplots()

        # Plot settings to match the animation's appearance
        ax.plot(np.rad2deg(self.sol[:, 0]), np.rad2deg(self.sol[:, 1]), color='navy', label="Phase Path",
                linewidth=2)

        # Set the labels, title, and grid
        ax.set_xlabel('θ1 / degrees')
        ax.set_ylabel('θ2 / degrees')
        ax.set_title('Phase Path', fontname='Courier New', fontsize=16)

        ax.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.7)
        plt.legend(loc='best')
        return fig

    def precompute_positions(self):
        """
        Precomputes and stores the positions of both pendulum bobs for each time step.

        This method calculates the (x, y) positions of the first and second pendulum bobs at each time step,
        using the provided initial conditions and system parameters. The positions are stored in a NumPy array
        as an instance attribute, which can be used for plotting and animation purposes, reducing the
        computational load at rendering time.
        """
        self.precomputed_positions = np.array(self._calculate_positions())

    def animate_pendulum(self, fig_width=700, fig_height=700, trace=False, static=False, appearance='light'):
        """
        Generates an animation for the double pendulum using precomputed positions.

        Parameters:
            fig_width (int): Default is 700 px
            fig_height (int): Default is 700 px
            trace (bool): If True, show the trace of the pendulum.
            static (bool): disables extra interactivity
            appearance (str): 'dark' for dark mode (default), 'light' for light mode.

        Raises:
            AttributeError: If `precompute_positions` has not been called before animation.

        Returns:
            A Plotly figure object containing the animation.
        """
        # Check if precomputed_positions has been calculated
        if not hasattr(self, 'precomputed_positions') or self.precomputed_positions is None:
            raise AttributeError("Precomputed positions must be calculated before animating. "
                                 "Please call 'precompute_positions' method first.")

        x_1, y_1, x_2, y_2 = self.precomputed_positions

        # Check appearance and set colors
        if appearance == 'dark':
            pendulum_color = 'rgba(255, 255, 255, 0.9)'  # White with slight transparency for visibility
            trace_color_theta1 = 'rgba(255, 165, 0, 0.6)'  # Soft orange with transparency for trace of P1
            trace_color_theta2 = 'rgba(0, 255, 0, 0.6)'  # Soft green with transparency for trace of P2
            background_color = 'rgb(17, 17, 17)'  # Very dark (almost black) for the plot background
            text_color = 'rgba(255, 255, 255, 0.9)'  # White text color for better visibility in dark mode
            grid_color = 'rgba(255, 255, 255, 0.3)'  # Light grey for grid lines

        elif appearance == 'light':
            pendulum_color = 'navy'  # Dark blue for better visibility against light background
            trace_color_theta1 = 'darkorange'  # Dark orange for a vivid contrast for trace of P1
            trace_color_theta2 = 'green'  # Dark green for trace of P2
            background_color = 'rgb(255, 255, 255)'  # White for the plot background
            text_color = 'rgb(0, 0, 0)'  # Black text color for better visibility in light mode
            grid_color = 'rgba(0, 0, 0, 0.1)'  # Light black (gray) for grid lines, with transparency for subtlety

        else:
            print("Invalid appearance setting. Please choose 'dark' or 'light'.")
            return None  # Exit the function if invalid appearance

        # Create figure with initial trace
        fig = go.Figure(
            data=[go.Scatter(
                x=[0, x_1[0], x_2[0]],
                y=[0, y_1[0], y_2[0]],
                mode='lines+markers',
                name='Pendulum',
                line=dict(width=2, color=pendulum_color),
                marker=dict(size=10, color=pendulum_color)
            )]
        )

        # If trace is True, add path traces
        if trace:
            path_1 = go.Scatter(
                x=x_1, y=y_1,
                mode='lines',
                name='Path of P1',
                line=dict(width=1, color=trace_color_theta1),
            )
            path_2 = go.Scatter(
                x=x_2, y=y_2,
                mode='lines',
                name='Path of P2',
                line=dict(width=1, color=trace_color_theta2),
            )
            fig.add_trace(path_1)
            fig.add_trace(path_2)

        # Calculate the max extent based on the precomputed positions
        max_extent = max(
            np.max(np.abs(x_1)),
            np.max(np.abs(y_1)),
            np.max(np.abs(x_2)),
            np.max(np.abs(y_2))
        )

        # Add padding to the max extent
        padding = 0.1 * max_extent  # 10% padding
        axis_range_with_padding = [-max_extent - padding, max_extent + padding]

        # Add frames to the animation
        step = 10
        frames = [go.Frame(data=[go.Scatter(x=[0, x_1[k], x_2[k]], y=[0, y_1[k], y_2[k]],
                                            mode='lines+markers',
                                            line=dict(width=2))])
                  for k in range(0, len(x_1), step)]  # Use a step to reduce the number of frames
        fig.frames = frames

        # Define the base layout configuration
        base_layout = dict(
            plot_bgcolor=background_color,
            paper_bgcolor=background_color,
            xaxis=dict(
                showgrid=True, gridwidth=1, gridcolor=grid_color,
                range=axis_range_with_padding,
                autorange=False, zeroline=False, tickcolor=text_color,
                tickfont=dict(size=12, color=text_color),
            ),
            yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor=grid_color,
                range=axis_range_with_padding,
                autorange=False, zeroline=False,
                scaleanchor='x', scaleratio=1,
                tickcolor=text_color,
                tickfont=dict(size=12, color=text_color),
            ),
            autosize=False,
            width=fig_width,
            height=fig_height,

        updatemenus=[{
            'type': 'buttons',
            'buttons': [
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {"frame": {"duration": 33, "redraw": True}, "fromcurrent": True,
                                "mode": "immediate",
                                'label': 'Play',
                                'font': {'size': 14, 'color': 'black'},
                                'bgcolor': 'lightblue'
                    }],
                )
            ],
            'direction': "left",
            'pad': {"r": 10, "t": 10},  # Adjust padding if needed
            'showactive': False,
            'type': 'buttons',
            'x': 0.05,  # Position for x
            'y': 0.95,  # Position for y,(the top of the figure)
            'xanchor': "left",
            'yanchor': "top"
        }],
        margin=dict(l=20, r=20, t=20, b=20),
        )
        # Update the layout based on the 'static' argument
        if static:
            static_updates = dict(
                xaxis_fixedrange=True,  # Disables horizontal zoom/pan
                yaxis_fixedrange=True,  # Disables vertical zoom/pan
                dragmode=False,         # Disables dragging
                showlegend=False        # Hides legend
            )
            fig.update_layout(**base_layout, **static_updates)
        else:
            fig.update_layout(**base_layout)

        return fig


class DoublePendulumExplorer(DoublePendulum):
    def __init__(self, parameters, time_vector, model, mechanical_energy,
                 theta2_range=(-np.pi, np.pi), **integrator_args):
        super().__init__(parameters, [0, 0, 0, 0], time_vector, model, **integrator_args)
        print("DoublePendulumExplorer initialized with base class.")
        self.mechanical_energy = mechanical_energy
        self.theta2_range = theta2_range

    def time_graph(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def phase_path(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def animate_pendulum(self, fig_width=700, fig_height=700, trace=False, static=False, appearance='light'):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def _generate_initial_conditions(self, step_size_degrees=0.5):
        """
        This method creates initial conditions where theta1 is fixed at 0, and theta2 varies from `self.theta2_range[0]`
        to `self.theta2_range[1]` in increments of `step_size_degrees`. The momentum values are fixed at 0.

        Parameters:
        ----------
        step_size_degrees : float, optional
            The increment step size in degrees for generating theta2 values. Defaults to 0.5 degrees.

        Returns:
        -------
        list of tuple
            A list of tuples representing the initial conditions for the simulations. Each tuple has the form (theta1, theta2, p1, p2),
            where theta1 is fixed at 0, theta2 varies according to the step size, and p1 and p2 are fixed at 0.

        """
        step_size_radians = np.deg2rad(step_size_degrees)
        theta2_min, theta2_max = self.theta2_range
        number_points = int((theta2_max - theta2_min) / step_size_radians)
        theta2_vals = np.linspace(theta2_min, theta2_max, number_points)
        initial_conditions = [(0, th2, 0, 0) for th2 in theta2_vals]

        return initial_conditions

    def _run_simulations(self, integrator=solve_ivp, batch_size=80, sleep_time=3, **integrator_args):
        """
        This method solves a system of ODEs for each set of initial conditions using the specified integrator, and stores
        the results in `self.initial_condition_data`. The simulations are executed in parallel to leverage multiple CPU
        cores and reduce computation time.

        Notes:
        -----
        - This method uses the `joblib` library to parallelize the simulations across all available CPU cores (`n_jobs=-1`).
          Each simulation is run independently, and results are collected asynchronously.
        - The method `_generate_initial_conditions` is used to generate the initial conditions for the simulations.
        - The method `_solve_ode` is called to solve the system of ODEs for each set of initial conditions. The results are
          stored in `self.initial_condition_data`.
        - If a simulation fails (e.g., due to numerical issues), the failure is caught and logged, and the simulation continues
          for the other initial conditions.

        """
        start_time = time.time()

        initial_conditions = self._generate_initial_conditions()

        num_simulations = len(initial_conditions)
        time_steps = self.time.size
        variables_per_step = 4

        # Initialize NumPy array to store all simulation data
        self.initial_condition_data = np.empty((num_simulations, time_steps, variables_per_step))

        def run_single_simulation(index, conditions):
            try:
                sol = self._solve_ode(integrator, initial_conditions=conditions, **integrator_args)
                return index, sol
            except Exception as e:
                print(f"Simulation {index} failed: {e}")
                return index, None

        # # Run all the simulations in parallel
        # results = Parallel(n_jobs=-1)(
        #     delayed(run_single_simulation)(index, cond) for index, cond in enumerate(initial_conditions))
        #
        # # Store the results in the initial_condition_data array
        # for index, sol in results:
        #     if sol is not None:
        #         self.initial_condition_data[index] = sol

        # Process simulations in batches
        for batch_start in range(0, num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, num_simulations)
            batch_conditions = initial_conditions[batch_start:batch_end]

            # Record batch start time
            batch_start_time = time.time()

            # Run the simulations in parallel for this batch
            results = Parallel(n_jobs=-3)(
                delayed(run_single_simulation)(batch_start + index, cond) for index, cond in
                enumerate(batch_conditions))

            # Store the results in the initial_condition_data array
            for index, sol in results:
                if sol is not None:
                    self.initial_condition_data[index] = sol

            # Record batch end time
            batch_end_time = time.time()
            batch_elapsed_time = batch_end_time - batch_start_time

            print(f"Batch {batch_start // batch_size + 1} of {num_simulations // batch_size} complete. "
                  f"Time taken: {batch_elapsed_time:.2f} seconds.")

            # Sleep between batches to allow my poor computer to cool down
            time.sleep(sleep_time)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Simulations Complete. Time taken: {elapsed_time:.2f} seconds.")

    def find_poincare_section(self, integrator=solve_ivp, **integrator_args):
        """
        Find the Poincaré section for the system based on the specified mechanical energy.

        This method ensures that the necessary simulation data is generated by calling _run_simulations
        if it has not already been done. It then computes the Poincaré section.
        """

        # Run simulations if they haven't been run yet
        if not hasattr(self, 'initial_condition_data') or self.initial_condition_data is None:
            self._run_simulations(integrator=integrator, **integrator_args)

        self.poincare_section_data = []

        for sim_idx, simulation in enumerate(self.initial_condition_data):
            theta1_values = simulation[:, 0]
            theta2_values = simulation[:, 1]
            p_theta_2_values = simulation[:, 3]

            poincare_points = []

            # Reset debug counter for each simulation
            debug_count = 0
            max_debug = 5

            tolerance = 1e-2  # Define a small tolerance value

            for i in range(1, len(theta1_values)):
                theta1_prev = theta1_values[i - 1]
                theta1_curr = theta1_values[i]

                # Check for traditional crossing or near-crossing using tolerance
                if (theta1_prev * theta1_curr < 0) or \
                        (abs(theta1_curr) < tolerance and theta1_prev * theta1_curr < 0) or \
                        (abs(theta1_prev) < tolerance and theta1_prev * theta1_curr < 0):

                    # Interpolation for the crossing point
                    ratio = -theta1_prev / (theta1_curr - theta1_prev)
                    theta2_interp = theta2_values[i - 1] + ratio * (theta2_values[i] - theta2_values[i - 1])
                    p_theta_2_interp = p_theta_2_values[i - 1] + ratio * (p_theta_2_values[i] - p_theta_2_values[i - 1])

                    # Optional debug statement to track detected crossings
                    # if debug_count < max_debug:
                    #     print(f"[DEBUG] Simulation {sim_idx}, Crossing Detected: theta2_interp = {theta2_interp}, p_theta_2_interp = {p_theta_2_interp}")
                    #     debug_count += 1

                    # Store as float32
                    poincare_points.append((np.float32(theta2_interp), np.float32(p_theta_2_interp)))

                    # Below is default float64
                    #poincare_points.append((theta2_interp, p_theta_2_interp))

            if poincare_points:
                self.poincare_section_data.append(poincare_points)
                #print(f"[DEBUG] Simulation {sim_idx}: {len(poincare_points)} Poincaré points detected.")
            else:
                print(f"[DEBUG] Simulation {sim_idx}: No Poincaré points detected.")

    def plot_poincare_map(self, special_angles_deg=None, xrange=(-np.pi, np.pi), yrange=None):
        """
        Plot the Poincaré section based on the computed data, with options to highlight special angles and restrict axes.

        Parameters:
        ----------
        special_angles_deg : list of float or None, optional
            A list of angles in degrees for which special trajectories should be highlighted in black.
            If None, no special trajectories are highlighted. Defaults to None.
        xrange : tuple of float, optional
            Limits for the x-axis in radians. Defaults to (-np.pi, np.pi).
        yrange : tuple of float or None, optional
            Limits for the y-axis. If None, the y-axis limits are set automatically. Defaults to None.
        """
        if not self.poincare_section_data:
            raise RuntimeError("No Poincaré data available. Run 'find_poincare_section' first.")

        plt.figure(figsize=(10, 10))

        # Create a colormap that contains as many colors as there are initial conditions
        colors = cm.viridis(np.linspace(0, 1, len(self.poincare_section_data)))

        # Plot each trajectory with a different color
        for i, poincare_points in enumerate(self.poincare_section_data):
            if poincare_points:
                theta2, p_theta_2 = zip(*poincare_points)
                plt.scatter(theta2, p_theta_2, s=0.05, color=colors[i])

        # Overlay special trajectories if special_angles_deg is provided
        if special_angles_deg is not None:
            special_indices = [(angle_deg + 180) / 0.5 for angle_deg in special_angles_deg]  # Calculate indices

            for i, index in enumerate(special_indices):
                index = int(index)
                if 0 <= index < len(self.poincare_section_data):
                    poincare_points = self.poincare_section_data[index]
                    if poincare_points:
                        theta2, p_theta_2 = zip(*poincare_points)
                        plt.scatter(theta2, p_theta_2, s=0.1, color='black')

        # Set x-axis limits
        plt.xlim(xrange)

        # Set y-axis limits if provided
        if yrange is not None:
            plt.ylim(yrange)

        # Set x-axis ticks
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(np.round(np.rad2deg(x)))}'))
        plt.xticks(np.linspace(xrange[0], xrange[1], 7))  # Control the number of ticks

        plt.xlabel(r'$\theta_2$ / degrees')
        plt.ylabel(r'$p_{\theta_2}$')
        plt.title(f'Poincaré Section with $\mathcal{{H}} \leq {self.mechanical_energy}$ $\\text{{J}}$\n'
                  f'{self.model.capitalize()} model')
        plt.grid(False)
        plt.show()


class DoublePendulumEnsemble(DoublePendulum):
    def __init__(self, parameters, time_vector, model, mechanical_energy,
                 theta1_range=(0, np.pi), theta2_range=(0, np.pi),
                 num_samples=10**3, random_seed=42, **integrator_args):
        super().__init__(parameters, [0, 0, 0, 0], time_vector, model, **integrator_args)
        print("DoublePendulumEnsemble initialized with base class.")
        self.mechanical_energy = mechanical_energy
        self.theta1_range = theta1_range
        self.theta2_range = theta2_range
        self.num_samples = num_samples
        self.random_seed = random_seed
        self.special_angles = [30, 45, 60, 90, 120, 135, 150]

        if random_seed is not None:
            np.random.seed(random_seed)

    def time_graph(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumEnsemble.")

    def phase_path(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumEnsemble.")

    def animate_pendulum(self, fig_width=700, fig_height=700, trace=False, static=False, appearance='light'):
        raise NotImplementedError("This method is not applicable for DoublePendulumEnsemble.")

    def calculate_potential_energy(self, theta1_val, theta2_val):
        """
        Calculate the potential energy of the double pendulum system relative to the datum where theta1 = 0 and theta2 = 0.
        """
        # Define symbolic variables
        if self.model == 'simple':
            V = -(m1 + m2) * g * l1 * sp.cos(theta1) - m2 * g * l2 * sp.cos(theta2)

        elif self.model == 'compound':
            V = -M1 * g * (l1 / 2) * sp.cos(theta1) - M2 * g * ((l1 * sp.cos(theta1)) + (l2 / 2) * sp.cos(theta2))

        else:
            raise ValueError("Model must be 'simple' or 'compound'")

        V = V.subs(self.parameters)
        V_subst = V.subs({theta1: theta1_val, theta2: theta2_val})
        # Calculate potential energy at theta1 = 0 and theta2 = 0 (datum)
        V_zero = V.subs({theta1: 0, theta2: 0})

        V_relative = V_subst - V_zero

        return V_relative

    def _generate_initial_conditions(self):
        """
        Generate initial conditions for the ensemble, including both special angle combinations
        and random samples within the specified ranges.

        Returns:
        -------
        list of tuple
            A list of tuples representing the initial conditions for the simulations.
            Each tuple has the form (theta1, theta2, p_theta_1, p_theta_2).
        """
        # Convert special angles from degrees to radians
        special_angles_rad = np.deg2rad(self.special_angles)

        # Generate special angle combinations
        special_conditions = []
        for th1 in special_angles_rad:
            for th2 in special_angles_rad:
                special_conditions.append((th1, th2, 0, 0))

        # Filter special conditions based on mechanical energy
        filtered_special_conditions = []
        for cond in special_conditions:
            theta1_val, theta2_val, _, _ = cond
            potential_energy = self.calculate_potential_energy(theta1_val, theta2_val)
            if potential_energy <= self.mechanical_energy:
                filtered_special_conditions.append(cond)

        # Store the number of special conditions as a class attribute
        self.num_special_conditions = len(filtered_special_conditions)
        print(f"Special Angle Combinations within energy limit: {self.num_special_conditions}")

        # Now calculate how many random samples are needed
        num_random_samples = self.num_samples - self.num_special_conditions
        random_conditions = []

        # Generate random conditions
        while len(random_conditions) < num_random_samples:
            random_theta1 = round(np.random.uniform(self.theta1_range[0], self.theta1_range[1]), 5)
            random_theta2 = round(np.random.uniform(self.theta2_range[0], self.theta2_range[1]), 5)
            potential_energy = self.calculate_potential_energy(random_theta1, random_theta2)

            if potential_energy <= self.mechanical_energy:
                random_conditions.append((random_theta1, random_theta2, 0, 0))
            #else:
                #print(f"Discarded condition: {(random_theta1, random_theta2, 0, 0)}")

        all_conditions = filtered_special_conditions + random_conditions
        print(f"Final length: {len(all_conditions)}")

        return all_conditions

    def _run_simulations(self, integrator=solve_ivp, sleep_time=5, **integrator_args):
        """
        This method solves a system of ODEs for each set of initial conditions using the specified integrator, and stores
        the results in `self.initial_condition_data`. The simulations are executed in parallel to leverage multiple CPU
        cores and reduce computation time.

        Notes:
        -----
        - This method uses the `joblib` library to parallelize the simulations across all available CPU cores (`n_jobs=-1`).
          Each simulation is run independently, and results are collected asynchronously.
        - The method `_generate_initial_conditions` is used to generate the initial conditions for the simulations.
        - The method `_solve_ode` is called to solve the system of ODEs for each set of initial conditions. The results are
          stored in `self.initial_condition_data`.
        - If a simulation fails (e.g., due to numerical issues), the failure is caught and logged, and the simulation continues
          for the other initial conditions.

        """
        start_time = time.time()

        initial_conditions = self._generate_initial_conditions()

        num_simulations = len(initial_conditions)
        time_steps = self.time.size
        variables_per_step = 4

        # Determine the batch size
        batch_size = max(1, int(self.num_samples / 10))
        batch_size = min(batch_size, 100)  # Ensure batch_size doesn't exceed 100

        # Initialize NumPy array to store all simulation data
        self.initial_condition_data = np.empty((num_simulations, time_steps, variables_per_step))

        def run_single_simulation(index, conditions):
            try:
                sol = self._solve_ode(integrator, initial_conditions=conditions, **integrator_args)
                return index, sol
            except Exception as e:
                print(f"Simulation {index} failed: {e}")
                return index, None

        # Process simulations in batches
        for batch_start in range(0, num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, num_simulations)
            batch_conditions = initial_conditions[batch_start:batch_end]

            # Record batch start time
            batch_start_time = time.time()

            # Run the simulations in parallel for this batch
            results = Parallel(n_jobs=-3)(
                delayed(run_single_simulation)(batch_start + index, cond) for index, cond in
                enumerate(batch_conditions))

            # Store the results in the initial_condition_data array
            for index, sol in results:
                if sol is not None:
                    if sol.shape[0] == time_steps:
                        self.initial_condition_data[index] = sol
                    else:
                        # print(
                        #     f"Warning: Simulation {index} returned {sol.shape[0]} time steps instead of {time_steps}.")
                        # # Optionally handle this case, for example by padding with NaNs:
                        padded_sol = np.full((time_steps, variables_per_step), np.nan)
                        padded_sol[:sol.shape[0], :] = sol
                        self.initial_condition_data[index] = padded_sol

            # Record batch end time
            batch_end_time = time.time()
            batch_elapsed_time = batch_end_time - batch_start_time

            print(f"Batch {batch_start // batch_size + 1} of {num_simulations // batch_size} complete. "
                  f"Time taken: {batch_elapsed_time:.2f} seconds.")

            # Sleep between batches to allow my poor computer to cool down
            time.sleep(sleep_time)

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Simulations Complete. Time taken: {elapsed_time:.2f} seconds.")


