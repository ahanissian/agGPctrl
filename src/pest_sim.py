# TODO: switch to xarrays


from dataclasses import dataclass

import numpy as np
import seaborn as sns
import catheat as ch
import math

import matplotlib.pyplot as plt



RECOVERED = 4       # treated with pesticide
SUSCEPTIBLE = 3     # no eggs or insects
LATENT = 2          # eggs present
INFECTED = 1        # insects present
DECEASED = 0        # deceased either by intentional culling or by disease


@dataclass
class SimulationConfig:
    """Stores simulation parameters for ABM pest simulation.

    Attributes:
        grid_size: tuple indicating dimension of grid representing the crop field.
        dt: hours per timestep
        ctrl_period: control action period (hours)
        duration: length of simulation rollout (hours)
        mean_latency_period: expected time a crop harbors disease but cannot spread it (hours)
        mean_transmission_period: expected time for an infected crop to transmit to neighbor (hours)
        mean_recovery_period: expected time to recover from infection (hours)
        daily_max_damage: Maximum % of damage experienced by crop while infected. In range [0,1].
        initial_infected: Number of crops latent with disease at start of simulation
        rng: Probability generator, if a fixed seed is desired.
    """
    grid_size: tuple
    dt: float = 4.0
    ctrl_period = 8.0
    duration: int = 50*24
    mean_latency_period: float = 4*24.0
    mean_transmission_period: float = 2.5*24.0
    mean_recovery_period: float = 14*24.0
    daily_max_damage: float = 0.1
    initial_infected: int = 5
    rng: np.random.Generator = np.random.default_rng(1)



class PestSimulation:
    """Stores parameters and state history of ABM pest simulation.

    This is a simplified agent based model representing the dispersion of blight throughout a crop
    initialization, a few crops are seeded with blight at uniformly sampled locations and enter
    field. The field is represented as a 2D grid with cells representing individual crops. Upon
    the LATENT state of infection, which is not detectable through observation. All other crops
    are initialized as SUSCEPTIBLE. A crop in the LATENT state has a chance to become INFECTED
    at every timestep, with the probability of becoming INFECTED being drawn from geometric
    distribution with expected value of N timesteps until infection, where N is the mean latency
    period divided by the timestep duration. Once INFECTED, a crop can spread disease to its nearest
    neighbors, making them LATENT with fixed probability at each timestep. An INFECTED crop may
    recover with a small fixed probability at each timestep. INFECTED crops also suffer damage at a
    specified rate. Crop healthis initialized at 1 and is decremented according to the damage rate
    until it falls below 0, at which point the plant is DECEASED.

    Attributes:
        config: SimulationConfig object containing parameters of simulation.
        n_steps: Number of timesteps in simulation rollout
        field_state: 2D array storing disease state of each cell e.g. 1 = Infected, 2 = Latent, etc.
        damage: 2D array storing damage state of each cell
        history: list containing accumulated field states over time.
        init_latent_cells: cells that are seeded with disesease at t=0
    """
    def __init__(self, config: SimulationConfig):
        """
        Initialize the simulation parameters and grid state.
        """
        self.config = config
        grid_size = self.config.grid_size
        self.n_steps = int(self.config.duration // self.config.dt)
        self.ctrl_interval = math.ceil(self.config.ctrl_period/self.config.dt)
        self.field_state = np.full(grid_size, SUSCEPTIBLE)
        self.damage = np.ones(self.config.grid_size)
        self.history = []

        # Initialize the grid with some infected cells
        self.init_latent_cells = np.random.choice(grid_size[0] * grid_size[1],
                                            self.config.initial_infected, replace=False)
        for pos in self.init_latent_cells:
            self.field_state[pos // grid_size[1],
                             pos % grid_size[1]] = LATENT
    
    def reset(self,keep_initial_positions = True):
        """
        Reset the simulation to its initial state.
        """
        self.field_state = np.full(self.config.grid_size, SUSCEPTIBLE)
        self.damage = np.ones(self.config.grid_size)
        self.history = []
    
        # Initialize the grid with some infected cells
        if keep_initial_positions:
            init_latent_cells = self.init_latent_cells
        else:
            init_latent_cells = np.random.choice(self.config.grid_size[0] * self.config.grid_size[1],
                                                self.config.initial_infected, replace=False)
        for pos in init_latent_cells:
            self.field_state[pos // self.config.grid_size[1],
                             pos % self.config.grid_size[1]] = LATENT

    def _update_field(self,field_state, action=[(-1,-1)]):
        """Rolls out simulation by one timestep.

        Args:
            field_state: 2D Array storing state of each gridcell at time T

        Returns:
            new_field: Field state array for time T+1
        """
        dt = self.config.dt
        grid_size = self.config.grid_size
        p_infection_geom = dt/self.config.mean_latency_period  # (mean tsteps until infection)^{-1}
        new_field = field_state.copy()

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                cell_state = field_state[i,j]

                apply_treatment = False
                if (i,j) in action:
                    apply_treatment = True

                if cell_state == INFECTED:
                    self._process_infected_cell(i,j, new_field, apply_treatment)
                elif cell_state == LATENT:
                    if apply_treatment:
                        new_field[i,j] = RECOVERED
                    elif self.config.rng.geometric(p_infection_geom) == 1:
                        new_field[i,j] = INFECTED
        return new_field
    
    def _process_infected_cell(self, i, j, new_field, apply_treatment):
        """
        Process an INFECTED cell and update its state in the new field.
        """
        dt = self.config.dt

        # apply damage since cell is infected
        damage_scale = self.config.daily_max_damage*dt/24 # max damage per timestep (x% per day)
        self.damage[i, j] -= np.random.rand() * damage_scale

        p_recovery_geom = dt/self.config.mean_recovery_period
        rng = self.config.rng

        # If the cell is still alive
        if self.damage[i, j] > 0:
            if apply_treatment:
                new_field[i, j] = RECOVERED
            elif rng.geometric(p_recovery_geom) == 1:  # Recovery condition
                new_field[i, j] = RECOVERED
            else:
                self._spread_infection(i, j, new_field)
        else:
            # If the cell is dead (damage <= 0)
            new_field[i, j] = DECEASED

    def _spread_infection(self, i, j, new_field):
        """
        Spread infection from an INFECTED cell to its neighbors.
        """
        rng = self.config.rng
        grid_size = self.config.grid_size
        # (mean tsteps until transmission)^{-1}
        p_transmission_geom = self.config.dt/self.config.mean_transmission_period 
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < grid_size[0] and 0 <= nj < grid_size[1]:
                if self.field_state[ni, nj] == SUSCEPTIBLE and rng.geometric(p_transmission_geom)==1:
                    new_field[ni, nj] = LATENT

    def simulate_open_loop(self):
        """
        Run the simulation and collect time-series data.
        """
        self.history = [self.field_state.copy()]
        for _ in range(self.n_steps):
            self.field_state = self._update_field(self.field_state)
            self.history.append(self.field_state.copy())
    
    def simulate_closed_loop(self,ctrl_callback):
        """
        Rollout the simulation in closed-loop with the provided controller.

        Args:
            ctrl_callback: function mapping history to list of tuples indicating gridpoints to treat
        """
        if not self.history:
            self.history = [self.field_state.copy()]
        for i in range(self.n_steps):
            if not (i % self.ctrl_interval):
                action = ctrl_callback(self.history)
                self.field_state = self._update_field(self.field_state, action)

    def plot_rollouts(self, time_points):
        """
        Plot the simulation rollouts at specified time points.
        """
        # Plot heatmap of infection density at various times
        fig, axes = plt.subplots(1, len(time_points), sharey=True, figsize=(20, 5))


        # Define a mapping from integers to category strings
        state_to_category = {
            RECOVERED: "Recovered",
            SUSCEPTIBLE: "Susceptible",
            LATENT: "Latent",
            INFECTED: "Infected",
            DECEASED: "Deceased"
        }

        # Create a copy of history with category strings
        history_with_categories = [
            np.vectorize(state_to_category.get)(state_grid) for state_grid in self.history
        ]

        magma_colors = sns.color_palette("magma",5)  # Extract 5 distinct colors for the states
        cmap = {
            "Recovered": magma_colors[3],
            "Susceptible": magma_colors[4],
            "Latent": magma_colors[2],
            "Infected": magma_colors[1],
            "Deceased": magma_colors[0]
        }

        for idx, t in enumerate(time_points):
            ax = axes[idx]
            ch.heatmap(history_with_categories[t], palette= "magma", cmap=cmap, ax=ax)
            ax.set_title(f"Step {t}")
        plt.tight_layout()
        plt.show()
