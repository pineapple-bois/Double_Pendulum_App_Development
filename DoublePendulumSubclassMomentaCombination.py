import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm
from matplotlib.ticker import FuncFormatter
from scipy.integrate import odeint, solve_ivp
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from joblib import Parallel, delayed
from AnalysisFunctions import *
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

    # def _solve_ode(self, integrator, initial_conditions=None, **integrator_args):
    #     """
    #     Solve the system of ODEs using the specified integrator.
    #
    #     Parameters:
    #     - integrator: The integrator function to use. Default is scipy's solve_ivp.
    #     - initial_conditions: The initial conditions for the ODE solver.
    #     - **integrator_args: Additional arguments specific to the chosen integrator.
    #     """
    #     if initial_conditions is None:
    #         initial_conditions = self.initial_conditions
    #
    #     if integrator == odeint:
    #         sol = odeint(self._system, initial_conditions, self.time, **integrator_args)
    #         return sol
    #     elif integrator == solve_ivp:
    #         t_span = (self.time[0], self.time[-1])
    #         sol = solve_ivp(lambda t, y: self._system(y, t), t_span, initial_conditions,
    #                         t_eval=self.time, **integrator_args)
    #         return sol.y.T  # Transpose
    #     else:
    #         raise ValueError("Unsupported integrator")

    def _solve_ode(self, integrator, initial_conditions=None, analyze=False, **integrator_args):
        """
        Solve the system of ODEs using the specified integrator and optionally analyze resource usage.

        Parameters:
        - integrator: The integrator function to use (e.g., solve_ivp).
        - initial_conditions: The initial conditions for the ODE solver.
        - analyze: Boolean, whether to analyze resource usage and save to a CSV file.
        - **integrator_args: Additional arguments specific to the chosen integrator.
        """
        if initial_conditions is None:
            initial_conditions = self.initial_conditions

        # Increment the file ticker only if analyzing
        csv_filename = None
        if analyze:
            self._file_ticker += 1
            csv_filename = f"simulation_data_{self._file_ticker}.csv"

        if integrator == odeint:
            sol = odeint(self._system, initial_conditions, self.time, **integrator_args)
            return sol

        elif integrator == solve_ivp:
            t_span = (self.time[0], self.time[-1])

            # Analyze the simulation if requested
            if analyze:
                analyse_computation_cost(
                    integrator=solve_ivp,
                    ode_system=lambda t, y: self._system(y, t),
                    initial_conditions=initial_conditions,
                    time_span=t_span,
                    csv_filename=csv_filename,
                    **integrator_args
                )

            sol = solve_ivp(lambda t, y: self._system(y, t), t_span, initial_conditions,
                            t_eval=self.time, **integrator_args)
            return sol.y.T  # Transpose

        else:
            raise ValueError("Unsupported integrator")

    def _calculate_positions(self):
        theta_1, theta_2 = self.sol[:, 0], self.sol[:, 1]

        l_1 = float(self.parameters[l1])
        l_2 = float(self.parameters[l2])

        x_1 = l_1 * np.sin(theta_1)
        y_1 = -l_1 * np.cos(theta_1)

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


class DoublePendulumExplorer(DoublePendulum):
    def __init__(self, parameters, time_vector, model, angle_range=(-np.pi, np.pi),
                 fixed_angle='theta1', step_size_degrees=0.5, **integrator_args):
        """
        Initialize the DoublePendulumExplorer class.

        Parameters:
        ----------
        parameters : dict
            The physical parameters of the double pendulum (e.g., masses, lengths).
        time_vector : numpy array
            The array of time points for the simulation.
        model : str
            The model of the double pendulum ('simple' or 'compound').
        angle_range : tuple, optional
            The range of the varying angle in radians. Defaults to (-pi, pi).
        fixed_angle : str, optional
            The angle to be held constant (either 'theta1' or 'theta2'). Defaults to 'theta1'.
        """
        super().__init__(parameters, [0, 0, 0, 0], time_vector, model, **integrator_args)
        print("DoublePendulumExplorer initialized with base class.")
        _file_ticker = 0
        self.angle_range = angle_range
        self.fixed_angle = fixed_angle
        self.step_size_degrees = step_size_degrees
        self.mechanical_energy = round(self._calculate_mechanical_energy(self.angle_range), 2)
        print(f"Mechanical energy: {self.mechanical_energy:.2f} J")

    def time_graph(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def phase_path(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def animate_pendulum(self, fig_width=700, fig_height=700, trace=False, static=False, appearance='light'):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def calculate_potential_energy(self, theta1_val, theta2_val):
        """
        Calculate the potential energy of the double pendulum system relative to the datum where theta1 = 0 and theta2 = 0.
        :param theta1_val:
        :param theta2_val:
        :return: float
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

    def _calculate_mechanical_energy(self, angle_range):
        """
        Dynamically calculate the mechanical energy based on the maximum potential energy within the specified angle range.
        :param angle_range: tuple
        :return: float
        """
        max_angle = max(abs(angle_range[0]), abs(angle_range[1]))
        if self.fixed_angle == 'theta1':
            return self.calculate_potential_energy(max_angle, 0)
        elif self.fixed_angle == 'theta2':
            return self.calculate_potential_energy(0, max_angle)
        else:
            raise ValueError("Invalid fixed_angle. Choose 'theta1' or 'theta2'.")

    def _generate_initial_conditions(self):
        """
        Generate initial conditions based on the fixed angle and angle range.

        Parameters:
        ----------
        step_size_degrees : float, optional
            The increment step size in degrees for generating angle values. Defaults to 0.5 degrees.

        Returns:
        -------
        list of tuple
            A list of tuples representing the initial conditions for the simulations.
        """
        step_size_radians = np.deg2rad(self.step_size_degrees)
        angle_min, angle_max = self.angle_range
        number_points = int((angle_max - angle_min) / step_size_radians)
        angle_vals = np.linspace(angle_min, angle_max, number_points)

        if self.fixed_angle == 'theta1':
            initial_conditions = [(th1, 0, 0, 0) for th1 in angle_vals]
        elif self.fixed_angle == 'theta2':
            initial_conditions = [(0, th2, 0, 0) for th2 in angle_vals]
        else:
            raise ValueError("Invalid fixed_angle. Choose 'theta1' or 'theta2'.")

        return initial_conditions

    def _run_simulations(self, integrator=solve_ivp, batch_size=80, sleep_time=10,
                         analyze=False, **integrator_args):
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
        - If `analyze` is True, resource usage will be tracked and saved to a CSV file.
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
                sol = self._solve_ode(integrator, initial_conditions=conditions, analyze=analyze, **integrator_args)
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

    def find_poincare_section(self, integrator=solve_ivp, fixed_angle=None, analyze=False, **integrator_args):
        """
        Find the Poincaré section for the system based on the mechanical energy.

        Parameters:
        ----------
        integrator : callable, optional
            The ODE solver to use for the simulations. Default is `scipy.integrate.solve_ivp`.
        fixed_angle : str, optional
            The angle to be held constant ('theta1' or 'theta2'). If None, the class attribute is used.
        **integrator_args : dict, optional
            Additional arguments to pass to the integrator.
        """
        if fixed_angle and fixed_angle != self.fixed_angle:
            print(f"Fixed angle changed from {self.fixed_angle} to {fixed_angle}. Re-running simulations.")
            self.fixed_angle = fixed_angle
            self.mechanical_energy = round(self._calculate_mechanical_energy(self.angle_range), 2)
            self._run_simulations(integrator=integrator, analyze=analyze, **integrator_args)
        elif not hasattr(self, 'initial_condition_data') or self.initial_condition_data is None:
            self._run_simulations(integrator=integrator, analyze=analyze, **integrator_args)

        # Guard clause to ensure initial_condition_data is properly set
        if self.initial_condition_data is None or len(self.initial_condition_data) == 0:
            raise RuntimeError("No simulation data available after running simulations.")

        self.poincare_section_data = []

        for sim_idx, simulation in enumerate(self.initial_condition_data):
            if self.fixed_angle == 'theta1':
                angle_values = simulation[:, 0]
                other_angle_values = simulation[:, 1]
                momentum_values = simulation[:, 2]
            elif self.fixed_angle == 'theta2':
                angle_values = simulation[:, 1]
                other_angle_values = simulation[:, 0]
                momentum_values = simulation[:, 3]
            else:
                raise ValueError("Invalid fixed_angle. Choose 'theta1' or 'theta2'.")

            poincare_points = []

            for i in range(1, len(other_angle_values)):
                angle_prev = other_angle_values[i - 1]
                angle_curr = other_angle_values[i]

                if angle_prev * angle_curr < 0:  # Detect crossings
                    ratio = -angle_prev / (angle_curr - angle_prev)
                    angle_interp = angle_values[i - 1] + ratio * (angle_values[i] - angle_values[i - 1])
                    momentum_interp = momentum_values[i - 1] + ratio * (momentum_values[i] - momentum_values[i - 1])

                    poincare_points.append((np.float32(angle_interp), np.float32(momentum_interp)))

            if poincare_points:
                self.poincare_section_data.append(poincare_points)
                # if sim_idx % 100 == 0:
                #     print(f"[DEBUG] Simulation {sim_idx}: {len(poincare_points)} Poincaré points detected.")
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
                theta1, p_theta_1 = zip(*poincare_points)
                plt.scatter(theta1, p_theta_1, s=0.05, color=colors[i])

        # Overlay special trajectories if special_angles_deg is provided
        if special_angles_deg is not None:
            special_indices = [(angle_deg + 180) / 0.5 for angle_deg in special_angles_deg]  # Calculate indices

            for i, index in enumerate(special_indices):
                index = int(index)
                if 0 <= index < len(self.poincare_section_data):
                    poincare_points = self.poincare_section_data[index]
                    if poincare_points:
                        theta1, p_theta_1 = zip(*poincare_points)
                        plt.scatter(theta1, p_theta_1, s=0.1, color='black')

        # Set x-axis limits
        plt.xlim(xrange)

        # Set y-axis limits if provided
        if yrange is not None:
            plt.ylim(yrange)

        # Set x-axis ticks
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(np.round(np.rad2deg(x)))}'))
        plt.xticks(np.linspace(xrange[0], xrange[1], 7))  # Control the number of ticks

        # Dynamically set axis labels based on the fixed angle
        if self.fixed_angle == 'theta1':
            xlabel = r'$\theta_2$ / degrees'
            ylabel = r'$p_{\theta_2}$'
            fixed_angle_label = r'$\theta_1$'
        elif self.fixed_angle == 'theta2':
            xlabel = r'$\theta_1$ / degrees'
            ylabel = r'$p_{\theta_1}$'
            fixed_angle_label = r'$\theta_2$'
        else:
            raise ValueError("Invalid fixed_angle. Choose 'theta1' or 'theta2'.")

        # Set the axis labels
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

        plt.title(f'Poincaré Section with $\mathcal{{H}} \leq {self.mechanical_energy:.2f}$ $\\text{{J}}$\n'
                  f'{self.model.capitalize()} model, {fixed_angle_label} constrained to zero')
        plt.grid(False)
        plt.show()

