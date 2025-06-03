# cs221-mechanics-path-search
This project provides a Python script (`mechanics.py`) for solving and analyzing one-dimensional classical mechanics problems, specifically:
1.  **Free Particle Motion**: A particle moving without external forces.
2.  **Free Fall Motion**: A particle moving under a uniform gravitational field.

The script uses various search algorithms to find trajectories that minimize the classical action (integral of the Lagrangian).

## Features

*   **Problem Definitions**: Implements `DiscreteFreeParticleProblem` and `FreeFallProblem` with configurable parameters.
*   **Search Algorithms**:
    *   Dynamic Programming
    *   Uniform Cost Search (modified to handle potentially negative step costs from the Lagrangian)
    *   A* Search (using an analytical action-to-go heuristic)
*   **Command-Line Interface**: Easy selection of problems and inference algorithms.
*   **Trajectory Plotting**: Visualizes position vs. time and velocity vs. time for the calculated trajectories.
*   **Performance Metrics**: When running all algorithms (`-i all`), it measures and plots:
    *   Runtime of each algorithm.
    *   Peak memory usage of each algorithm.
*   **Analytical Comparison**: Calculates and displays the Root Mean Squared Error (RMSE) between the algorithm's solution and the analytical solution for both position and velocity.
*   **Plot Saving**: Automatically saves generated plots as PNG files in the script's directory, overwriting previous files with the same name.

## Prerequisites

*   Python 3.8 or newer.
*   `pip` (Python package installer).

## Setup and Installation

1.  **Clone the repository (or download the files):**
    ```bash
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Execute the script from the command line:
```bash
python mechanics.py [options]
```

### Command-Line Arguments:

*   `-p, --problem {free_fall, free_particle}`:
    *   Specifies the type of problem to solve.
    *   Default: `free_fall`
*   `-i, --inference {dp, ucs, astar, all}`:
    *   Specifies the inference algorithm to use.
    *   `dp`: Dynamic Programming
    *   `ucs`: Uniform Cost Search
    *   `astar`: A* Search
    *   `all` (default): Runs all three algorithms sequentially, generates comparative plots, and performance metrics.

### Examples:

*   Solve the free fall problem using Dynamic Programming:
    ```bash
    python mechanics.py -p free_fall -i dp
    ```
*   Solve the free particle problem using all algorithms and compare them:
    ```bash
    python mechanics.py -p free_particle -i all
    ```
*   Run the default (free fall problem with all algorithms):
    ```bash
    python mechanics.py
    ```

## Output
*   **Console**: Detailed trajectory information (time, position, velocity), total cost (action), RMSE values against analytical solutions, and performance metrics (runtime, memory).
*   **Plots**: Interactive Matplotlib windows showing position and velocity trajectories.
*   **Saved PNGs**: All generated plots are automatically saved as PNG files in the directory where the script is run (e.g., `solution_Free_Fall_Dynamic_Programming.png`, `all_solutions_Free_Fall.png`, `performance_metrics_Free_Fall.png`).
