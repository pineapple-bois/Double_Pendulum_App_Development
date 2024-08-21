# Extension of the [Double Pendulum App](https://github.com/pineapple-bois/Double_Pendulum_App/tree/main). Development of the `Chaos` page


1. #### Extend the `DoublePendulum` class to `DoublePendulumExplorer` capable of integrating a range of initial conditions. 
   - Specifically; $\theta_2 \in [-\pi, \pi], \text{step}=0.5^{\circ}$ and $t \in [0, 120]\text{s}$
2. #### Write a data dictionary in JSON format of all angles, velocities and positions 
   - Currently these use the Lagrangian and are 2 x 9Gb!
3. #### Host this data as a PostgreSQL DB on a cloud server (aiming for v.cheap/free)
4. #### Iterate over and slice the DB to plot Poincaré sections and Lyapunov exponents

----

### All of the above will form the basis of a new page for the web app. Instead of deriving a double pendulum system 'on-the-fly', we will pull data from the database with simple slicing queries.

### Development Directory Structure

```
Double_Pendulum_App_Development/
├── JSONdata/
├── Notebooks/
│   ├── DevelopmentSubClass.ipynb
│   ├── JSONTest.ipynb
│   └── PoincareSections.ipynb
├── PendulumModels/
│   ├── DoublePendulumHamiltonian.py
│   ├── DoublePendulumLagrangian.py
├── .gitattributes
├── DoublePendulumSubclass.py
├── DoublePendulumSubclassDev.py
├── DoublePendulumSubclassMomenta.py
├── MathFunctions.py
├── README.md
└── requirements.txt
```

1. ### [`DoublePendulumSubclass.py`](pyscripts/DoublePendulumSubclass.py)

   The `DoublePendulumExplorer` subclass extends the functionality of the `DoublePendulum` class to explore a range of initial conditions for a double pendulum system. It focuses on how varying the initial angle $\theta_2$ affects the system's dynamics, and it provides tools for visualising Poincaré sections and other dynamic behaviors.

   &nbsp;
     - **Exploration of Initial Conditions**: Vary $\theta_2$ while keeping other initial conditions fixed to see how different initial angles affect the dynamics.
     - **Poincaré Sections**: Calculate and visualize Poincaré sections to gain insights into the system's phase space structure and identify periodic or chaotic behavior.
     - **Simulations and Data Structures**: Run multiple simulations and organise the results in a structured format for easy analysis and visualisation.

   
### 20/08/24: The Class was refactored to [`DoublePendulumSubclassMomenta.py`](DoublePendulumSubclassMomenta.py) 

This change focuses on energy conservation as well as utilising $\text{p}_{\theta_1}$ and $\text{p}_{\theta_2}$

#### Refactoring Steps

1. **Introduction of Potential Energy Calculations:**
   - We added the ability to calculate the potential energy of the double pendulum system. This was crucial for performing energy-based analysis, such as finding Poincaré sections based on energy levels.
   - The `_calculate_potential_energy` method was introduced, which accounts for both the 'simple' and 'compound' models, allowing the class to compute potential energy relative to a defined zero-potential reference point.

2. **Energy-Based Poincaré Sections:**
   - The `find_poincare_section` method was updated to focus on energy-based crossings, where the potential energy at a given crossing is compared to a specified maximum potential energy level. This method was refactored to interpolate values at the crossing points and only record them if the energy condition is met.


### The new class has been tested in [`PoincareSections.ipynb`](Notebooks/PoincareSections.ipynb) and the initial Poincaré plots look promising

---

2. ### [`DevelopmentSubClass.ipynb`](Notebooks/DevelopmentSubClass.ipynb)
   - Have started writing the base methods for the subclass.
   - The data dictionaries appear to be quite good!
   - The Poincaré sections are really not what we are looking for...

----

3. ### [`JSONTest.ipynb`](Notebooks/JSONTest.ipynb)
   - Reading in the JSON data using Pandas (Will maybe swap for Polars once DB launched)

----

4. ### Using the Hamiltonian for Analysing Chaos and Periodic Orbits

**Advantages of Hamiltonian Mechanics in Chaos and Periodic Orbits:**
1. **Phase Space Analysis:**
   - Hamiltonian mechanics naturally leads to the analysis of the system in phase space (coordinates and momenta), which is crucial for studying chaotic behaviour and periodic orbits.
   - Trajectories in phase space can reveal fixed points, periodic orbits, and chaotic regions.

2. **Energy Conservation:**
   - In conservative systems, the Hamiltonian is a conserved quantity (total energy). This restriction of analysis to constant energy surfaces, simplifying the study of dynamics.

3. **Symplectic Structure:**
   - The Hamiltonian framework preserves the symplectic structure, which is important in the study of dynamical systems and chaos theory.

#### Steps to Map Trajectories and Analyse Chaos

1. **Derive the Hamiltonian:**
   - Ensure that the Hamiltonian is correctly derived from the Lagrangian. In your case, you’ve already done this for the double pendulum.

2. **Compute Hamilton's Equations:**
   - Use Hamilton's equations to obtain the equations of motion:
     $\dot{q}_i = \frac{\partial H}{\partial p_i}$, $\dot{p}_i = -\frac{\partial H}{\partial q_i}$ where $\mathbf{q}=(\theta_1, \theta_2)$
   - These equations describe how the coordinates $q_i$ and momenta $p_i$ evolve over time.

3. **Phase Space Trajectories:**
   - Solve Hamilton’s equations numerically to obtain trajectories in phase space. Use Runge-Kutta integration.
   - Visualise the trajectories to identify periodic orbits, fixed points, and chaotic behaviour.

4. **Poincaré Sections:**
   - Create Poincaré sections (intersections of phase space trajectories with a lower-dimensional subspace) to visualise the structure of the system. This will help in identifying periodic orbits and chaotic regions.

5. **Lyapunov Exponents:**
   - Calculate Lyapunov exponents to quantify the sensitivity of the system to initial conditions. Positive Lyapunov exponents indicate chaos.

6. **Energy Conservation:**
   - Ensure that energy is conserved in numerical simulations. Deviations may indicate numerical errors.

7. **Bifurcation Diagrams:**
   - Vary a system parameter (e.g., length or mass) and observe changes in the system's behaviour. Plot bifurcation diagrams to identify transitions between periodic and chaotic behaviour.

8. **Periodic Orbits and Stability:**
   - Identify periodic orbits and analyse their stability. Unstable periodic orbits can be indicative of chaotic regions.

### Further Reading and Tools:
- **Books:**
  - "Nonlinear Dynamics and Chaos" by Steven Strogatz
  - "Chaos: An Introduction to Dynamical Systems" by Kathleen T. Alligood, Tim D. Sauer, and James A. Yorke
- **Software Tools:**
  - Check out [`chaospy`](https://chaospy.readthedocs.io/en/master/)
- **Papers:**
  - [Numerical analysis of a Double Pendulum System - Dartmouth](https://math.dartmouth.edu/archive/m53f09/public_html/proj/Roja_writeup.pdf)
  - [The double pendulum: a numerical study - A M Calvão and T J P Penna](https://iopscience-iop-org.libezproxy.open.ac.uk/article/10.1088/0143-0807/36/4/045018)

----
