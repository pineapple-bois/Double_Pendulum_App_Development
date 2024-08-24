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
        with open("termination_data/raw_termination_data_three.csv", 'a', newline='') as csvfile:
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
                        max_angle_limit = 2 * (2 * np.pi)   # Allow 15 loops - May need to be tweaked
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
                        # Return only the valid portion and pad with zeros
                        valid_sol = sol.y.T[:len(sol.t)]
                        padded_sol = np.zeros((self.time.size, valid_sol.shape[1]))
                        padded_sol[:valid_sol.shape[0], :] = valid_sol
                        return padded_sol

                    # Check for NaNs or Infs
                    if np.any(np.isnan(sol.y)) or np.any(np.isinf(sol.y)):
                        writer.writerow({'initial_conditions': initial_conditions,
                                         'termination_time': sol.t[-1],
                                         'termination_reason': 'NaN/Inf detected'})
                        return None # Return None for invalid data

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

    @staticmethod
    def _get_appearance_settings(appearance):
        if appearance == 'dark':
            return {
                'pendulum_color': 'rgba(255, 255, 255, 0.9)',
                'trace_color_theta1': 'rgba(255, 165, 0, 0.6)',
                'trace_color_theta2': 'rgba(0, 255, 0, 0.6)',
                'background_color': 'rgb(17, 17, 17)',
                'text_color': 'rgba(255, 255, 255, 0.9)',
                'grid_color': 'rgba(255, 255, 255, 0.3)'
            }
        elif appearance == 'light':
            return {
                'pendulum_color': 'navy',
                'trace_color_theta1': 'darkorange',
                'trace_color_theta2': 'green',
                'background_color': 'rgb(255, 255, 255)',
                'text_color': 'rgb(0, 0, 0)',
                'grid_color': 'rgba(0, 0, 0, 0.1)'
            }
        else:
            raise ValueError("Invalid appearance setting. Please choose 'dark' or 'light'.")

    @staticmethod
    def _calculate_axis_range(x_1, y_1, x_2, y_2):
        max_extent = max(
            np.max(np.abs(x_1)),
            np.max(np.abs(y_1)),
            np.max(np.abs(x_2)),
            np.max(np.abs(y_2))
        )
        padding = 0.1 * max_extent  # 10% padding
        return [-max_extent - padding, max_extent + padding]

    @staticmethod
    def _create_base_layout(axis_range, appearance_settings, fig_width, fig_height, static):
        base_layout = dict(
            plot_bgcolor=appearance_settings['background_color'],
            paper_bgcolor=appearance_settings['background_color'],
            xaxis=dict(
                showgrid=True, gridwidth=1, gridcolor=appearance_settings['grid_color'],
                range=axis_range,
                autorange=False, zeroline=False, tickcolor=appearance_settings['text_color'],
                tickfont=dict(size=12, color=appearance_settings['text_color']),
            ),
            yaxis=dict(
                showgrid=True, gridwidth=1, gridcolor=appearance_settings['grid_color'],
                range=axis_range,
                autorange=False, zeroline=False,
                scaleanchor='x', scaleratio=1,
                tickcolor=appearance_settings['text_color'],
                tickfont=dict(size=12, color=appearance_settings['text_color']),
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
                        args=[None, {"frame": {"duration": 33, "redraw": True},
                                    "fromcurrent": True,
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
                'x': 0.05,  # Position for x
                'y': 0.95,  # Position for y,(the top of the figure)
                'xanchor': "left",
                'yanchor': "top"
            }],

            margin=dict(l=20, r=20, t=20, b=20),
        )

        if static:
            static_updates = dict(
                xaxis_fixedrange=True,  # Disables horizontal zoom/pan
                yaxis_fixedrange=True,  # Disables vertical zoom/pan
                dragmode=False,         # Disables dragging
                showlegend=False        # Hides legend
            )
            base_layout.update(static_updates)

        return base_layout

    @staticmethod
    def _create_trace(x, y, trace_name, color, mode='lines', width=1):
        return go.Scatter(
            x=x, y=y,
            mode=mode,
            name=trace_name,
            line=dict(width=width, color=color)
        )

    @staticmethod
    def _create_animation_frames(x_1, y_1, x_2, y_2, step=10):
        return [
            go.Frame(data=[go.Scatter(x=[0, x_1[k], x_2[k]], y=[0, y_1[k], y_2[k]],
                                      mode='lines+markers', line=dict(width=2))])
            for k in range(0, len(x_1), step)
        ]

    def animate_pendulum(self, fig_width=700, fig_height=700, trace=True, static=False, appearance='light'):
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
        if not hasattr(self, 'precomputed_positions') or self.precomputed_positions is None:
            raise AttributeError("Precomputed positions must be calculated before animating. "
                                 "Please call 'precompute_positions' method first.")

        x_1, y_1, x_2, y_2 = self.precomputed_positions
        appearance_settings = self._get_appearance_settings(appearance)
        axis_range = self._calculate_axis_range(x_1, y_1, x_2, y_2)

        fig = go.Figure(
            data=[self._create_trace([0, x_1[0], x_2[0]], [0, y_1[0], y_2[0]], 'Pendulum',
                                     appearance_settings['pendulum_color'], mode='lines+markers', width=2)]
        )

        if trace:
            fig.add_trace(self._create_trace(x_1, y_1, 'Path of P1', appearance_settings['trace_color_theta1']))
            fig.add_trace(self._create_trace(x_2, y_2, 'Path of P2', appearance_settings['trace_color_theta2']))

        fig.frames = self._create_animation_frames(x_1, y_1, x_2, y_2)
        fig.update_layout(self._create_base_layout(axis_range, appearance_settings, fig_width, fig_height, static))

        return fig


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
        start_time = time.time()
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

            # Add filtering conditions
            if random_theta1 > np.deg2rad(150):
                continue

            if random_theta2 > np.deg2rad(120) and random_theta1 > np.deg2rad(100):
                continue

            # Potentially add more conditions here

            potential_energy = self.calculate_potential_energy(random_theta1, random_theta2)

            if potential_energy <= self.mechanical_energy:
                random_conditions.append((random_theta1, random_theta2, 0, 0))
            #else:
                #print(f"Discarded condition: {(random_theta1, random_theta2, 0, 0)}")

        all_conditions = filtered_special_conditions + random_conditions

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Final length: {len(all_conditions)}. Time taken: {elapsed_time:.2f} seconds.")

        return all_conditions

    def _run_simulations(self, integrator=solve_ivp, sleep_time=10, **integrator_args):
        """
        Run simulations for each set of generated initial conditions.

        This method solves a system of ODEs for each set of initial conditions generated by the
        `_generate_initial_conditions` method using the specified integrator. The simulations are
        executed in parallel to take advantage of multiple CPU cores, reducing computation time.
        The results of each simulation are stored in the `self.initial_condition_data` array.

        Parameters:
        ----------
        integrator : callable, optional
            The ODE solver to use for the simulations. Default is `scipy.integrate.solve_ivp`.
        sleep_time : int or float, optional
            Time to sleep between batches of simulations (in seconds) to prevent system overload.
            Default is 5 seconds.
        **integrator_args : dict, optional
            Additional arguments to pass to the integrator.

        Notes:
        -----
        - The method uses the `joblib` library to parallelize the simulations across available CPU
          cores (`n_jobs=-3` indicates that all but two CPU cores are used). Each simulation runs
          independently, and results are collected asynchronously.
        - The initial conditions for the simulations are generated by the `_generate_initial_conditions`
          method.
        - Simulations are processed in batches to manage memory usage effectively. Batch size is
          determined as one-tenth of the total number of samples, with a maximum batch size of 100.
        - If a simulation fails (e.g., due to numerical issues), the failure is logged, and the
          simulation continues for the other initial conditions. Failed simulations result in `None`
          values in the `self.initial_condition_data` array.
        - If the number of time steps in a simulation result is less than expected, the result is
          padded with NaNs to maintain consistent array dimensions.

        """
        start_time = time.time()

        initial_conditions = self._generate_initial_conditions()
        print(f"\nRunning Simulations:")

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

            for index, sol in results:
                if sol is not None:
                    self.initial_condition_data[index] = sol
                else:
                    print(f"Simulation {index} returned invalid data.")

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

    def find_poincare_section(self, integrator=solve_ivp, crossing_point='theta1', **integrator_args):
        """
        Find the Poincaré section for the system based on the specified mechanical energy.

        Parameters:
        ----------
        integrator : callable, optional
            The ODE solver to use for the simulations. Default is `scipy.integrate.solve_ivp`.
        crossing_point : str, optional
            The variable to use for detecting crossings ('theta1' or 'theta2'). Default is 'theta1'.
        **integrator_args : dict, optional
            Additional arguments to pass to the integrator.
        """

        # Run simulations if they haven't been run yet
        if not hasattr(self, 'initial_condition_data') or self.initial_condition_data is None:
            self._run_simulations(integrator=integrator, **integrator_args)

        print(f"\nFinding Poincaré points:")
        self.poincare_section_data = []

        num_simulations = self.initial_condition_data.shape[0]

        # Determine the batch size dynamically based on index 0 of initial_condition_data
        batch_size = max(1, int(len(self.initial_condition_data[0]) / 10))
        batch_size = min(batch_size, 100)  # Ensure batch_size doesn't exceed 100

        # Calculate the total number of batches
        total_batches = num_simulations // batch_size

        def process_integration_batch(simulation, crossing_point):
            theta1_values = simulation[:, 0]
            theta2_values = simulation[:, 1]
            p_theta_1_values = simulation[:, 2]
            p_theta_2_values = simulation[:, 3]

            poincare_points = []

            for i in range(1, len(theta1_values)):
                if crossing_point == 'theta1':
                    crossing_prev = theta1_values[i - 1]
                    crossing_curr = theta1_values[i]
                    other_var_values = theta2_values
                    conjugate_momentum_values = p_theta_1_values
                elif crossing_point == 'theta2':
                    crossing_prev = theta2_values[i - 1]
                    crossing_curr = theta2_values[i]
                    other_var_values = theta1_values
                    conjugate_momentum_values = p_theta_2_values
                else:
                    raise ValueError("Invalid crossing_point. Choose 'theta1' or 'theta2'.")

                # Check for traditional crossing
                if crossing_prev * crossing_curr < 0:
                    # Interpolation for the crossing point
                    ratio = -crossing_prev / (crossing_curr - crossing_prev)
                    other_var_interp = other_var_values[i - 1] + ratio * (
                                    other_var_values[i] - other_var_values[i - 1])
                    momentum_interp = conjugate_momentum_values[i - 1] + ratio * (
                                    conjugate_momentum_values[i] - conjugate_momentum_values[i - 1])

                    # Store as float32
                    poincare_points.append((np.float32(other_var_interp), np.float32(momentum_interp)))

            return poincare_points

        # Process simulations in batches
        for batch_start in range(0, num_simulations, batch_size):
            batch_end = min(batch_start + batch_size, num_simulations)
            batch_simulations = self.initial_condition_data[batch_start:batch_end]

            # Record batch start time
            batch_start_time = time.time()

            # Parallel processing for the current batch
            results = Parallel(n_jobs=-1)(
                delayed(process_integration_batch)(simulation, crossing_point) for simulation in batch_simulations
            )

            # Collect results
            for sim_idx, poincare_points in enumerate(results, start=batch_start):
                if poincare_points:
                    self.poincare_section_data.append(poincare_points)
                    # Print every 100th index
                    if sim_idx % 100 == 0:
                        print(f"[DEBUG] Simulation {sim_idx}: {len(poincare_points)} Poincaré points detected.")
                else:
                    print(f"[DEBUG] Simulation {sim_idx}: No Poincaré points detected.")

            # Record batch end time
            batch_end_time = time.time()
            batch_elapsed_time = batch_end_time - batch_start_time

            current_batch = (batch_start // batch_size) + 1
            print(f"Batch {current_batch} of {total_batches} complete. Time taken: {batch_elapsed_time:.2f} seconds.")

    def plot_poincare_map(self, xrange=(-np.pi, np.pi), yrange=None):
        """
        Plot the Poincaré section based on the computed data, with options to restrict axes.

        Parameters:
        ----------
        xrange : tuple of float, optional
            Limits for the x-axis in radians. Defaults to (-np.pi, np.pi).
        yrange : tuple of float or None, optional
            Limits for the y-axis. If None, the y-axis limits are set automatically. Defaults to None.
        """
        if not self.poincare_section_data or len(self.poincare_section_data) == 0:
            print("No Poincaré data available. Ensure simulations are run before plotting.")
            return

        plt.figure(figsize=(10, 10))

        # Create a colormap that contains as many colors as there are initial conditions
        colors = cm.viridis(np.linspace(0, 1, len(self.poincare_section_data)))

        # Plot each trajectory with a different color
        for i, poincare_points in enumerate(self.poincare_section_data):
            if poincare_points:
                theta2, p_theta_2 = zip(*poincare_points)
                plt.scatter(theta2, p_theta_2, s=0.05, color=colors[i])

        # Set x-axis limits
        plt.xlim(xrange)

        # Set y-axis limits if provided
        if yrange is not None:
            plt.ylim(yrange)

        # Set x-axis ticks
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(np.round(np.rad2deg(x)))}'))
        plt.xticks(np.linspace(xrange[0], xrange[1], 7))  # Control the number of ticks

        plt.xlabel(r'$\theta$ / degrees')
        plt.ylabel(r'$\operatorname{p}_{\theta}$')
        plt.title(f'Poincaré Section with $\mathcal{{H}} \leq {self.mechanical_energy}$ $\\text{{J}}$\n'
                  f'{self.model.capitalize()} model')
        plt.grid(False)

        plt.show()




