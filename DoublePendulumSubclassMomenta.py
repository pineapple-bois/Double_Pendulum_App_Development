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

        if integrator == odeint:
            sol = odeint(self._system, initial_conditions, self.time, **integrator_args)
        elif integrator == solve_ivp:
            t_span = (self.time[0], self.time[-1])
            sol = solve_ivp(lambda t, y: self._system(y, t), t_span, initial_conditions,
                            t_eval=self.time, **integrator_args)
            sol = sol.y.T  # Transpose
        else:
            raise ValueError("Unsupported integrator")

        return sol

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
                 theta1_cross_section=0, theta2_range=(-np.pi, np.pi), **integrator_args):
        super().__init__(parameters, [0, 0, 0, 0], time_vector, model, **integrator_args)
        print("DoublePendulumExplorer initialized with base class.")
        _, _, V = form_lagrangian(model)     # Returns Lagrangian, Kinetic Energy, Potential Energy
        self.V = V.subs(parameters)
        self.mechanical_energy = mechanical_energy
        self.theta1_cross_section = theta1_cross_section
        self.theta2_range = theta2_range
        self._data_ready = False # Flag to track if simulation data is ready for structure computation

    def time_graph(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def phase_path(self):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def animate_pendulum(self, fig_width=700, fig_height=700, trace=False, static=False, appearance='light'):
        raise NotImplementedError("This method is not applicable for DoublePendulumExplorer.")

    def _calculate_potential_energy(self, theta1_val, theta2_val, model='simple'):
        # Perform substitutions for the potential energy (V)
        if model == 'simple':
            V_zero = -(m1 + m2) * g * l1 - m2 * g * l2
        elif model == 'compound':
            V_zero = -M1 * g * (l1 / 2) - M2 * g * ((l1 + l2) / 2)
        else:
            raise ValueError("Model must be 'simple' or 'compound'")
        V_zero_numeric = V_zero.subs(self.parameters)

        V_numeric = self.V.subs({
            theta1: theta1_val,
            theta2: theta2_val
        })

        V_relative = V_numeric - V_zero_numeric
        # Evaluate the expression numerically
        return float(V_relative)

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

    def _run_simulations(self, integrator=solve_ivp):
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
                sol = self._solve_ode(integrator, initial_conditions=conditions)
            except Exception as e:
                print(f"Simulation {index} failed: {e}")
                return index, None
            return index, sol

        results = Parallel(n_jobs=-1)(
            delayed(run_single_simulation)(index, cond) for index, cond in enumerate(initial_conditions))

        for index, sol in results:
            if sol is not None:
                self.initial_condition_data[index] = sol

        end_time = time.time()
        elapsed_time = end_time - start_time

        print(f"Simulations Complete. Time taken: {elapsed_time:.2f} seconds.")

    def find_poincare_section(self, energy_tolerance=1e-2, integrator=solve_ivp):
        """
        Find the Poincaré section for the system based on the specified mechanical energy.

        This method ensures that the necessary simulation data is generated by calling _run_simulations
        if it has not already been done. It then computes the Poincaré section.
        """

        # Run simulations if they haven't been run yet
        if not hasattr(self, 'initial_condition_data') or self.initial_condition_data is None:
            self._run_simulations(integrator=integrator)

        self.poincare_section_data = []

        # Pre-calculate the mechanical energy tolerance for quick comparisons
        max_mechanical_energy = self.mechanical_energy + energy_tolerance

        for simulation in self.initial_condition_data:
            theta1_values = simulation[:, 0]
            theta2_values = simulation[:, 1]
            p_theta_2_values = simulation[:, 3]

            poincare_points = []

            for i in range(1, len(theta1_values)):
                theta1_prev = theta1_values[i - 1]
                theta1_curr = theta1_values[i]

                # Check for crossing through the theta1 cross-section (theta1 = 0)
                if (theta1_prev - self.theta1_cross_section) * (theta1_curr - self.theta1_cross_section) < 0:
                    # Interpolation for the crossing point
                    ratio = -theta1_prev / (theta1_curr - theta1_prev)
                    theta2_interp = theta2_values[i - 1] + ratio * (theta2_values[i] - theta2_values[i - 1])
                    p_theta_2_interp = p_theta_2_values[i - 1] + ratio * (p_theta_2_values[i] - p_theta_2_values[i - 1])

                    # Calculate potential energy at the crossing point
                    potential_energy = self._calculate_potential_energy(self.theta1_cross_section, theta2_interp,
                                                                        self.model)

                    # Record if the potential energy is within the allowed range
                    if potential_energy <= max_mechanical_energy:
                        poincare_points.append((theta2_interp, p_theta_2_interp))

            if poincare_points:
                self.poincare_section_data.append(poincare_points)

    def plot_poincare_map(self):
        """
        Plot the Poincaré section based on the computed data.
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

        plt.xlim(-np.pi, np.pi)
        plt.xlabel(r'$\theta_2$')
        plt.ylabel(r'$p_{\theta_2}$')
        plt.title(f'Poincaré Section at $\mathcal{{H}} = {self.mechanical_energy}$ $\\text{{J}}$\n'
                  f'{self.model.capitalize()} model')
        plt.grid(False)
        plt.show()
