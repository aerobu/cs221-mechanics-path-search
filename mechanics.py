import heapq
import matplotlib.pyplot as plt
import argparse
import time
import tracemalloc
import numpy as np
from typing import List, Tuple, Optional, Any, Dict, Callable # For type hints

"""
Solves 1D classical mechanics problems (Free Particle, Free Fall) using various
search algorithms to find trajectories that minimize the classical action.

Features:
- Implements Dynamic Programming, Uniform Cost Search (handling negative costs),
  and A* search algorithms.
- Provides command-line interface for selecting problems and algorithms.
- Visualizes trajectories (position and velocity vs. time).
- Measures and compares runtime and memory usage of algorithms.
- Calculates Root Mean Squared Error (RMSE) against analytical solutions.
- Saves all generated plots as PNG files.

Usage:
  python mechanics.py -p <problem_type> -i <inference_type>

  <problem_type>: 'free_fall' (default) or 'free_particle'
  <inference_type>: 'dp', 'ucs', 'astar', or 'all' (default)
"""

class DiscreteFreeParticleProblem:
    """
    Models a discrete free particle problem.
    The particle moves in 1D without external forces. The goal is to find a
    trajectory (sequence of positions and velocities over time) from an initial
    state (q0, t0) to a final state (qf, tf) that minimizes the action,
    S = integral(0.5 * m * v^2 dt).
    """
    def __init__(self):
        self.m = 1 
        self.q0 = 50
        self.qf = 750
        self.t0 = 0
        self.tf = 100
        self.dt = 1
        self.qmin = 0
        self.qmax = 3000
        self.maxSpeed = 10
        self.possibleVelocities = [v for v in range(-self.maxSpeed, self.maxSpeed+1)]

    def startState(self):
        """Returns the initial state (time, position) of the particle."""
        return (self.t0, self.q0) 

    def isEnd(self, state):
        """
        Checks if the given state (time, position) is the target final state.
        Args:
            state: A tuple (time, position).
        Returns:
            True if the state is the end state, False otherwise.
        """
        if state:
            return state[0] == self.tf and state[1] == self.qf
        else:
            return False

    def succAndCost(self, state): 
        """
        Generates successor states and the cost (action contribution) for each.
        Args:
            state: The current state (time, position).
        Returns:
            A list of tuples (action, nextState, cost_step), where action is
            the velocity chosen for the step.
        """
        t_current, q = state
        if t_current >= self.tf:
            return []
        t_next = t_current + self.dt
        retList = []
        for v in self.possibleVelocities:
            cost = self._calculate_delta_s(v)
            q_next = round(q + v * self.dt)
            if q_next > self.qmax or q_next < self.qmin:
                continue
            nextState = (t_next, q_next)
            retList.append((v, nextState, cost)) 
        return retList

    def _calculate_delta_s(self, v):
        """
        Calculates the contribution to the action for a time step dt
        with constant velocity v.
        S_step = L * dt = (0.5 * m * v^2) * dt.
        Args:
            v: The velocity during the time step.
        Returns:
            The cost (action contribution) for the step.
        """
        return 0.5 * self.m * (v**2) * self.dt 

class FreeFallProblem:
    """
    Models a particle in free fall under uniform gravity.
    The particle moves in 1D. The goal is to find a trajectory from an initial
    state (q0, t0) to a final state (qf, tf) that minimizes the action,
    S = integral((0.5 * m * v^2 - m * g * y) dt).
    """
    def __init__(self):
        self.m = 1 
        self.g = 10 
        self.q0 = 20
        self.qf = 0
        self.t0 = 0
        self.tf = 20
        self.dt = 1
        self.qmin = 0
        self.qmax = 3000
        self.maxSpeed = 20
        self.possibleVelocities = [v for v in range(-self.maxSpeed, self.maxSpeed+1)]

    def startState(self):
        """
        Returns the initial state (time, position, velocity).
        Velocity is initially None, to be determined by the first action.
        """
        return (self.t0, self.q0, None) 

    def isEnd(self, state):
        """
        Checks if the given state (time, position, velocity) is the target final state.
        Only time and position are checked against qf, tf.
        Args:
            state: A tuple (time, position, velocity).
        Returns:
            True if the state is the end state, False otherwise.
        """
        if state:
            return state[0] == self.tf and state[1] == self.qf
        else:
            return False
    def succAndCost(self, state): 
        """
        Generates successor states and the cost (action contribution) for each.
        Velocity changes are constrained to the current velocity or +/- 1 unit.
        Args:
            state: The current state (time, position, current_velocity).
        Returns:
            A list of tuples (action, nextState, cost_step), where action is
            the velocity chosen for the step.
        """
        t_current, q, qdot = state
        if t_current >= self.tf:
            return []
        t_next = t_current + self.dt
        retList = []

        possibleVelocities = []
        if qdot is not None:
            currVelIdx = self.possibleVelocities.index(qdot)
            possibleVelocities.append(self.possibleVelocities[currVelIdx])
            if currVelIdx == 0:
                possibleVelocities.append(self.possibleVelocities[1])
            elif currVelIdx == len(self.possibleVelocities) - 1:
                possibleVelocities.append(self.possibleVelocities[-2])
            else:
                possibleVelocities.append(self.possibleVelocities[currVelIdx-1])
                possibleVelocities.append(self.possibleVelocities[currVelIdx+1])
        else:
            possibleVelocities = self.possibleVelocities

        for v in possibleVelocities:
            q_next = round(q + v * self.dt)
            cost = self._calculate_delta_s(q, v)
            if q_next > self.qmax or q_next < self.qmin:
                continue
            nextState = (t_next, q_next, v)
            retList.append((v, nextState, cost)) 
        return retList

    def _calculate_delta_s(self, y, v):
        """
        Calculates the contribution to the action for a time step dt.
        S_step = L * dt = (0.5 * m * v^2 - m * g * y) * dt.
        Args:
            y: The position (height) at the start of the time step.
            v: The velocity during the time step.
        Returns:
            The cost (action contribution) for the step.
        """
        return (0.5 * self.m * (v**2) - self.m * self.g * y) * self.dt 

def printSolution(solution):
    """
    Prints the total cost and history of a solution found by a search algorithm.
    Args:
        solution: A tuple (totalCost, history_list).
    """
    totalCost, history = solution
    print('totalCost:', totalCost)
    # Limit printing history for very long paths to avoid excessive console output
    if len(history) > 50:
        print(f"  (Displaying first 20 and last 20 steps of {len(history)} total steps)")
        for item in history[:20]:
            print(item)
        print("  ...")
        for item in history[-20:]:
            print(item)
    else:
        for item in history:
            print(item)

# --- Analytical Functions (copied from analytical.py to avoid import side-effects) ---
# These functions provide the exact solutions for position and velocity
# for the Free Particle and Free Fall problems, used for RMSE calculation.

def q_analytical_fp(t, q0, qf, t0, tf):
    """Analytical position for a free particle."""
    # Ensure t0 != tf to avoid division by zero if called with such values
    if tf == t0:
        return np.full_like(t, q0) if q0 == qf else np.full_like(t, np.nan) # Or handle error appropriately
    return q0 + (qf - q0) / (tf - t0) * (t - t0)

def v_analytical_fp(t, q0, qf, t0, tf):
    """Analytical velocity for a free particle (constant)."""
    if tf == t0:
        return np.zeros_like(t) if q0 == qf else np.full_like(t, np.nan)
    val = (qf - q0) / (tf - t0)
    if isinstance(t, np.ndarray):
        return np.full_like(t, val)
    return val

def get_v0_ff(q0, qf, t0, tf, g):
    """
    Calculates the initial velocity required for an object in free fall
    to travel from (q0, t0) to (qf, tf) under constant gravity g.
    """
    if tf == t0: # Avoid division by zero
        return 0.0 if q0 == qf else np.nan # Or some indicator of impossibility/undefined
    return (qf - q0) / (tf - t0) + 0.5 * g * (tf - t0)

def q_analytical_ff(t, q0, t0, v0, g):
    """
    Analytical position for free fall motion.
    q(t) = q0 + v0*(t-t0) - 0.5*g*(t-t0)^2
    """
    return q0 + v0 * (t - t0) - 0.5 * g * (t - t0)**2

def v_analytical_ff(t, t0, v0, g):
    """
    Analytical velocity for free fall motion.
    v(t) = v0 - g*(t-t0)
    """
    return v0 - g * (t - t0)

# --- RMSE Calculation against Analytical Solution ---
def calculate_and_print_rmse_against_analytical(history, problem_instance, problem_name_str, inference_method_str):
    """
    Calculates and prints the Root Mean Squared Error (RMSE) for position and
    velocity between the algorithm's solution and the analytical solution.
    Args:
        history: The solution path from the search algorithm.
        problem_instance: The problem definition instance.
        problem_name_str: String name of the problem (e.g., "Free Fall").
        inference_method_str: String name of the inference method (e.g., "A* Search").
    """
    if not history:
        return

    t_algo, q_algo, v_algo = _extract_plot_data(history, problem_instance)
    if not t_algo: # No data points
        return

    t_np = np.array(t_algo)
    q_algo_np = np.array(q_algo)
    v_algo_np = np.array(v_algo)

    q_exact = np.array([])
    v_exact = np.array([])

    print(f"\n--- RMSE vs Analytical for {inference_method_str} on {problem_name_str} ---")

    if problem_name_str == "Free Particle":
        if isinstance(problem_instance, DiscreteFreeParticleProblem):
            q_exact = q_analytical_fp(t_np, problem_instance.q0, problem_instance.qf, problem_instance.t0, problem_instance.tf)
            v_exact = v_analytical_fp(t_np, problem_instance.q0, problem_instance.qf, problem_instance.t0, problem_instance.tf)
    elif problem_name_str == "Free Fall":
        if isinstance(problem_instance, FreeFallProblem):
            v0_analytical = get_v0_ff(problem_instance.q0, problem_instance.qf, problem_instance.t0, problem_instance.tf, problem_instance.g)
            q_exact = q_analytical_ff(t_np, problem_instance.q0, problem_instance.t0, v0_analytical, problem_instance.g)
            v_exact = v_analytical_ff(t_np, problem_instance.t0, v0_analytical, problem_instance.g)

    if q_exact.size > 0 and q_exact.size == q_algo_np.size:
        # Filter out NaNs that might result from analytical functions if t_algo contains problematic values (e.g., t > tf for some reason)
        # or if problem parameters lead to undefined analytical solutions (e.g. t0=tf)
        valid_indices = ~np.isnan(q_exact) & ~np.isnan(q_algo_np)
        if np.any(valid_indices):
            rmse_q = np.sqrt(np.mean((q_algo_np[valid_indices] - q_exact[valid_indices])**2))
            print(f"Position RMSE: {rmse_q:.4f}")
        else:
            print("Position RMSE: Not calculable (no valid comparable points).")
    else:
        print("Position RMSE: Not calculable (analytical or algorithm data missing/mismatched).")

    if v_exact.size > 0 and v_exact.size == v_algo_np.size:
        valid_indices_v = ~np.isnan(v_exact) & ~np.isnan(v_algo_np)
        if np.any(valid_indices_v):
            rmse_v = np.sqrt(np.mean((v_algo_np[valid_indices_v] - v_exact[valid_indices_v])**2))
            print(f"Velocity RMSE: {rmse_v:.4f}")
        else:
            print("Velocity RMSE: Not calculable (no valid comparable points).")
    else:
        print("Velocity RMSE: Not calculable (analytical or algorithm data missing/mismatched).")
    print("----------------------------------------------------")

def _extract_plot_data(history, problem_instance):
    """
    Helper function to extract time, position, and velocity values from
    a solution history for plotting.
    Args:
        history: The solution path from a search algorithm. Each item is
                 (action, state, cost_step).
        problem_instance: The problem definition instance.
    Returns:
        A tuple of three lists: (t_values, q_values, v_values).
    """
    if not history:
        return [], [], []

    t_vals = []
    q_vals = []
    v_vals = []
    is_ffp = isinstance(problem_instance, FreeFallProblem)

    for i, (action, state, cost_step) in enumerate(history):
        t_vals.append(state[0])
        q_vals.append(state[1])

        if is_ffp:
            if i == 0:  
                if len(history) > 1 and history[1][0] is not None: 
                    v_vals.append(history[1][0])
                elif state[2] is not None: 
                    v_vals.append(state[2])
                else: 
                    v_vals.append(0)
            else: 
                v_vals.append(state[2] if state[2] is not None else 0)
        else: 
            if action is not None:
                v_vals.append(action)
            else:  
                if i == 0 and len(history) > 1 and history[1][0] is not None: 
                    v_vals.append(history[1][0])
                else:
                    v_vals.append(0) 
    return t_vals, q_vals, v_vals

def plotSolution(history, problem_instance, problem_name_str, inference_method_str):
    """
    Plots the position vs. time and velocity vs. time for a given solution.
    Saves the plot to a PNG file.
    Args:
        history: The solution path from the search algorithm.
        problem_instance: The problem definition instance.
        problem_name_str: String name of the problem.
        inference_method_str: String name of the inference method.
    """
    if not history:
        print("Cannot plot empty history.")
        return

    t_vals = []
    q_vals = []
    v_vals = []    
    t_vals, q_vals, v_vals = _extract_plot_data(history, problem_instance)
    if not t_vals: 
        print("No data to plot after extraction.")
        return

    fig, axs = plt.subplots(2, 1, figsize=(10, 8)) 

    fig.suptitle(f"Solution for {problem_name_str} using {inference_method_str}", fontsize=14)

    axs[0].plot(t_vals, q_vals, marker='.', linestyle='-')
    axs[0].set_title("Position as a Function of Time")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("q(t) or x(t)")
    axs[0].grid(True)

    axs[1].plot(t_vals, v_vals, marker='.', linestyle='-', color='orange')
    axs[1].set_title("Velocity as a Function of Time")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("v(t)")
    axs[1].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    filename = f"solution_{problem_name_str.replace(' ', '_')}_{inference_method_str.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()
    plt.close(fig) # Close the figure window

def plot_all_solutions(all_results_data):
    """
    Generates and displays a combined plot showing the position and velocity
    trajectories for all algorithms run in 'all' mode.
    Saves the plot to a PNG file.
    Args:
        all_results_data: A list of dictionaries, where each dictionary contains
                          the results (history, problem_instance, names, metrics) for one algorithm.
    """
    if not all_results_data:
        print("No results to plot for 'all' mode.")
        return

    num_results = len(all_results_data)
    fig, axs = plt.subplots(num_results, 2, figsize=(12, 4 * num_results), squeeze=False)
    
    problem_name_overall = all_results_data[0]["problem_name"]
    fig.suptitle(f"Comparison of Algorithms for {problem_name_overall}", fontsize=16)

    for idx, result_data in enumerate(all_results_data):
        history = result_data["history"]
        problem_instance = result_data["problem_instance"]
        inference_name = result_data["inference_name"]

        t_vals, q_vals, v_vals = _extract_plot_data(history, problem_instance)

        axs[idx, 0].plot(t_vals, q_vals, marker='.', linestyle='-')
        axs[idx, 0].set_title(f"{inference_name} - Position vs. Time")
        axs[idx, 0].set_xlabel("t")
        axs[idx, 0].set_ylabel("q(t) or x(t)")
        axs[idx, 0].grid(True)

        axs[idx, 1].plot(t_vals, v_vals, marker='.', linestyle='-', color='orange')
        axs[idx, 1].set_title(f"{inference_name} - Velocity vs. Time")
        axs[idx, 1].set_xlabel("t")
        axs[idx, 1].set_ylabel("v(t)")
        axs[idx, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    filename = f"all_solutions_{problem_name_overall.replace(' ', '_')}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()
    plt.close(fig) # Close the figure window

def plot_performance_metrics(all_results_data):
    """
    Generates and displays a bar chart comparing the runtime and peak memory usage
    of all algorithms run in 'all' mode.
    Saves the plot to a PNG file.
    Args:
        all_results_data: A list of dictionaries containing performance metrics.
    """
    if not all_results_data:
        print("No performance data to plot.")
        return

    names = [data["inference_name"] for data in all_results_data]
    runtimes = [data["runtime"] for data in all_results_data]
    memory_usages_kib = [data["memory_usage"] / 1024 for data in all_results_data]

    x = np.arange(len(names))  
    width = 0.35  

    fig, ax1 = plt.subplots(figsize=(12, 7))

    color_runtime = 'tab:blue'
    ax1.set_xlabel('Algorithm')
    ax1.set_ylabel('Runtime (seconds)', color=color_runtime)
    bars_runtime = ax1.bar(x - width/2, runtimes, width, label='Runtime (s)', color=color_runtime)
    ax1.tick_params(axis='y', labelcolor=color_runtime)
    ax1.grid(True, axis='y', linestyle='--', alpha=0.7, which='major')

    ax2 = ax1.twinx()
    color_memory = 'tab:red'
    ax2.set_ylabel('Peak Memory Usage (KiB)', color=color_memory)
    bars_memory = ax2.bar(x + width/2, memory_usages_kib, width, label='Memory (KiB)', color=color_memory)
    ax2.tick_params(axis='y', labelcolor=color_memory)

    fig.suptitle('Algorithm Performance Comparison', fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names)
    
    fig.legend(handles=[bars_runtime, bars_memory], loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2)

    fig.tight_layout(rect=[0, 0.1, 1, 0.95]) 
    problem_name_for_filename = all_results_data[0]["problem_name"].replace(' ', '_')
    filename = f"performance_metrics_{problem_name_for_filename}.png"
    plt.savefig(filename)
    print(f"Plot saved to {filename}")
    plt.show()
    plt.close(fig) # Close the figure window

def dynamicProgramming(problem):
    """
    Solves the given search problem using dynamic programming.
    It computes the minimum cost (action) to reach the end state from any
    intermediate state by recursively calculating future costs.
    Assumes the problem has an optimal substructure and overlapping subproblems,
    and that the state space forms a Directed Acyclic Graph (DAG) or that
    costs are such that cycles are not an issue for finite termination.
    Args:
        problem: An instance of a problem class (e.g., FreeFallProblem)
                 with startState(), isEnd(), and succAndCost() methods.
    Returns:
        A tuple (totalCost, history_list), where totalCost is the minimum
        action, and history_list details the optimal path.
    """
    cache: Dict[Any, Tuple[float, Optional[Any], Optional[Any], Optional[float]]] = {}  # state => (futureCost, action, newState, cost_of_step)
    def futureCost(state):
        if problem.isEnd(state):
            return 0
        if state in cache:  
            return cache[state][0]
        nextStates = [] 
        for action, newState, cost in problem.succAndCost(state):
            fc = futureCost(newState)
            nextState = (cost + futureCost(newState), action, newState, cost)
            nextStates.append(nextState)
        if nextStates:
            cache[state] = min(nextStates)
        else:
            cache[state] = (float('inf'), None, None, None)
        return cache[state][0]

    state = problem.startState()
    totalCost = futureCost(state)

    history = []
    history.append((None, state, None)) # (action_leading_to_state, state, cost_of_that_action)
    while not problem.isEnd(state):
        print(state)
        cost_from_state, action, newState, cost = cache[state]
        if action is None or newState is None : # No path found from this state
            # This might happen if the problem is unsolvable from an intermediate state
            # or if the goal was reached but totalCost was inf.
            # If totalCost is inf, history should reflect no solution.
            # If totalCost is finite, this implies a state with no valid next action in cache.
            if totalCost == float('inf'): # If no solution was ever found.
                return (float('inf'), []) # Return empty history for no solution
            # If we are here, it means futureCost(state) was finite, but cache[state] is bad.
            # This indicates an issue in logic or an unsolvable subproblem not caught by futureCost.
            print(f"Warning: DP path reconstruction stuck at state {state}. Cache: {cache.get(state)}")
            # Return partial history if a cost was found, otherwise indicate failure.
            # For DP, if totalCost is finite, a full path to goal should exist in cache.
            # This break implies an issue if totalCost was not 'inf'.
            break 
        history.append((action, newState, cost))
        state = newState

    return (totalCost, history)


def uniformCostSearch(problem):
    """
    Placeholder for a standard Uniform Cost Search.
    Note: This version of UCS is typically for non-negative costs.
    The `uniformCostSearch_with_negative_costs` is used in this script
    due to the nature of the Lagrangian.
    """
    # This is a basic UCS, not the one used for negative costs (PriorityQueueSPFA)
    # For the project, uniformCostSearch_with_negative_costs is the relevant one.
    raise NotImplementedError("Standard UCS not fully implemented/used; use uniformCostSearch_with_negative_costs.")

class PriorityQueue:
    """
    A standard priority queue for algorithms like Uniform Cost Search
    where states are marked 'DONE' once expanded (suitable for non-negative costs).
    """
    def __init__(self):
        self.DONE = -100000
        self.heap = []
        self.priorities = {}  

    def update(self, state, newPriority: float) -> bool:
        oldPriority = self.priorities.get(state)
        if oldPriority is None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if self.priorities[state] == self.DONE:
                continue
            self.priorities[state] = self.DONE
            return state, priority
        return None, None

class PriorityQueueSPFA:
    """
    A priority queue adapted for algorithms like SPFA (Shortest Path Faster Algorithm)
    or UCS variants that need to handle negative edge costs and re-open states.
    It does not mark states as 'DONE' upon removal, allowing for updates if a shorter path is found later.
    """
    def __init__(self):
        self.heap = []
        self.priorities = {}  

    def update(self, state, newPriority: float) -> bool:
        oldPriority = self.priorities.get(state)
        if oldPriority is None or newPriority < oldPriority:
            self.priorities[state] = newPriority
            heapq.heappush(self.heap, (newPriority, state))
            return True
        return False

    def removeMin(self):
        while len(self.heap) > 0:
            priority, state = heapq.heappop(self.heap)
            if state not in self.priorities or priority > self.priorities[state]:
                continue
            return state, priority
        
        return None, None # Heap is empty

def uniformCostSearch_with_negative_costs(problem):
    """
    Solves the given search problem using a Uniform Cost Search variant
    that can handle negative step costs (actions). This is similar to the
    Bellman-Ford or SPFA approach where states can be re-added to the frontier
    if a cheaper path is found. It finds the path with the minimum cumulative cost.
    Args:
        problem: An instance of a problem class (e.g., FreeFallProblem)
                 with startState(), isEnd(), and succAndCost() methods.
    Returns:
        A tuple (totalCost, history_list), where totalCost is the minimum
        action, and history_list details the optimal path. If no path is
        found, totalCost is float('inf') and history_list is empty.
    """
    # Type hints for clarity
    frontier = PriorityQueueSPFA() 
    backpointers = {} 

    startState = problem.startState()
    frontier.update(startState, 0)

    min_cost_to_goal = float('inf')
    final_goal_state = None

    while True:
        state, pastCost = frontier.removeMin()

        if state is None: 
            break 

        if problem.isEnd(state):
            if pastCost < min_cost_to_goal: 
                min_cost_to_goal = pastCost
                final_goal_state = state


        for action, newState, step_cost in problem.succAndCost(state):
            new_cost_to_newState = pastCost + step_cost
            
            if frontier.update(newState, new_cost_to_newState):
                backpointers[newState] = (action, state, step_cost)

    if final_goal_state is None and problem.isEnd(startState) and 0 < min_cost_to_goal :
        if 0 < min_cost_to_goal: 
             min_cost_to_goal = 0
             final_goal_state = startState


    if final_goal_state is not None:
        history = []
        curr_s = final_goal_state
        path_segments_reversed = []

        if curr_s == startState and not backpointers: 
             path_segments_reversed.append((None, startState, 0))
        else:
            while curr_s != startState :
                if curr_s not in backpointers:
                    break 
                action, prev_s, cost_s = backpointers[curr_s]
                path_segments_reversed.append((action, curr_s, cost_s))
                curr_s = prev_s
        
        plot_history = [(None, startState, None)] 
        for seg_action, seg_state, seg_cost in reversed(path_segments_reversed):
            plot_history.append((seg_action, seg_state, seg_cost))
        
        if not path_segments_reversed and final_goal_state == startState:
             plot_history = [(None, startState, None)]


        return (min_cost_to_goal, plot_history)
    
    return (float('inf'), [])

class AStarPriorityQueue:
    """
    A priority queue specifically for A* search.
    It stores states prioritized by their f-value (g_cost + heuristic).
    Handles updates if a lower f-value is found for an existing state.
    """
    def __init__(self):
        self.heap = []
        self.f_values = {} 

    def update(self, state, new_f_value: float) -> bool:
        current_best_f = self.f_values.get(state)
        if current_best_f is None or new_f_value < current_best_f:
            self.f_values[state] = new_f_value
            heapq.heappush(self.heap, (new_f_value, state))
            return True
        return False

    def removeMin(self): 
        while len(self.heap) > 0:
            f_value, state = heapq.heappop(self.heap)
            if state not in self.f_values or f_value > self.f_values[state]:
                continue
            return f_value, state 
        return None, None

def a_star_search(problem, heuristic_func):
    """
    Solves the given search problem using the A* search algorithm.
    A* explores paths by prioritizing states with lower f-values, where
    f(state) = g(state) + h(state).
    g(state) is the actual cost from the start state to 'state'.
    h(state) is the estimated heuristic cost from 'state' to the goal.
    Args:
        problem: An instance of a problem class.
        heuristic_func: A function that takes a state and the problem instance,
                        and returns an estimated cost-to-go.
    Returns:
        A tuple (totalCost, history_list), where totalCost is the minimum
        action found, and history_list details the optimal path.
    """
    frontier = AStarPriorityQueue()
    g_costs = {} 
    backpointers = {}
    startState = problem.startState()
    g_costs[startState] = 0.0
    h_start = heuristic_func(startState, problem) 
    f_start = g_costs[startState] + h_start
    frontier.update(startState, f_start)
    min_total_g_cost_to_goal = float('inf')
    final_goal_state_for_path = None

    while True:
        popped_f_value, currentState = frontier.removeMin()
        if currentState is None:
            break
        current_g_cost = g_costs[currentState]

        if popped_f_value >= min_total_g_cost_to_goal and heuristic_func(currentState, problem) >=0 :
             continue 

        if problem.isEnd(currentState):
            if current_g_cost < min_total_g_cost_to_goal:
                min_total_g_cost_to_goal = current_g_cost
                final_goal_state_for_path = currentState

        for action, newState, step_cost in problem.succAndCost(currentState):
            new_g_cost_to_newState = current_g_cost + step_cost

            if new_g_cost_to_newState < g_costs.get(newState, float('inf')):
                g_costs[newState] = new_g_cost_to_newState
                backpointers[newState] = (action, currentState, step_cost)
                
                h_newState = heuristic_func(newState, problem)
                f_newState = new_g_cost_to_newState + h_newState
                frontier.update(newState, f_newState)
                
    if final_goal_state_for_path is not None:
        plot_history = [] 
        path_segments_reversed = []
        curr_s = final_goal_state_for_path

        if curr_s == startState and not backpointers.get(curr_s) :
             pass 
        else:
            while curr_s != startState:
                if curr_s not in backpointers:
                    break 
                action, prev_s, cost_s = backpointers[curr_s]
                path_segments_reversed.append((action, curr_s, cost_s))
                curr_s = prev_s
        
        plot_history.append((None, startState, None))
        for seg_action, seg_state, seg_cost in reversed(path_segments_reversed):
            plot_history.append((seg_action, seg_state, seg_cost))
        
        return (min_total_g_cost_to_goal, plot_history)
        
    return (float('inf'), [])

def zero_heuristic(state, problem_instance):
    """
    A trivial heuristic that always returns 0.
    When used with A*, it makes A* behave like Uniform Cost Search.
    Args:
        state: The current state.
        problem_instance: The problem definition instance.
    Returns:
        0.0
    """
    return 0.0

def analytical_action_to_go_heuristic(state, problem_instance):
    """
    A heuristic for A* search that calculates the analytical action (integral of Lagrangian)
    to go from the current state to the goal state, assuming an unconstrained path.
    Args:
        state: The current state (time, position, [velocity]).
        problem_instance: The problem definition instance.
    Returns:
        The calculated heuristic value (estimated action-to-go).
    """
    if problem_instance.isEnd(state):
        return 0.0

    tf = problem_instance.tf
    qf = problem_instance.qf
    m = problem_instance.m

    if isinstance(problem_instance, FreeFallProblem):
        t_curr, q_curr, _ = state  
    elif isinstance(problem_instance, DiscreteFreeParticleProblem):
        t_curr, q_curr = state     

    if t_curr >= tf:
        return float('inf')

    Th = tf - t_curr

    if isinstance(problem_instance, FreeFallProblem):
        g = problem_instance.g 
        v0_h = (qf - q_curr) / Th + 0.5 * g * Th
        integral_T_term = 0.5 * m * (v0_h**2 * Th - g * v0_h * Th**2 + (1.0/3.0) * g**2 * Th**3)
        integral_V_component = -m * g * (q_curr * Th + 0.5 * v0_h * Th**2 - (1.0/6.0) * g * Th**3)
        
        heuristic_action = integral_T_term + integral_V_component
        return heuristic_action

    elif isinstance(problem_instance, DiscreteFreeParticleProblem):
        v_opt = (qf - q_curr) / Th 
        heuristic_action = 0.5 * m * (v_opt**2) * Th
        return heuristic_action
    

if __name__ == "__main__":
    """
    Main execution block for running the mechanics solver from the command line.
    """
    parser = argparse.ArgumentParser(description="Original Mechanics Solver (mechanics.py). Solves physics problems using various search algorithms.")
    parser.add_argument("-p", "--problem", type=str, choices=["free_fall", "free_particle"], default="free_fall",
                        help="Type of problem to solve (default: free_fall)")
    parser.add_argument("-i", "--inference", type=str, choices=["dp", "ucs", "astar", "all"], default="all",
                        help="Inference algorithm to use. 'all' runs dp, ucs, and astar sequentially (default: all)")
    
    args = parser.parse_args()

    problem_instance_orig = None
    problem_name_str_orig = ""

    if args.problem == "free_fall":
        problem_instance_orig = FreeFallProblem()
        problem_name_str_orig = "Free Fall"
        print(f"Selected Problem (mechanics.py): {problem_name_str_orig} (q0={problem_instance_orig.q0}, qf={problem_instance_orig.qf}, tf={problem_instance_orig.tf})")
    elif args.problem == "free_particle":
        problem_instance_orig = DiscreteFreeParticleProblem()
        problem_name_str_orig = "Free Particle"
        print(f"Selected Problem (mechanics.py): {problem_name_str_orig} (q0={problem_instance_orig.q0}, qf={problem_instance_orig.qf}, tf={problem_instance_orig.tf})")
    
    if problem_instance_orig is None:
        print("Error: Problem instance could not be created.")
        exit(1)
    
    print(f"Start state (mechanics.py): {problem_instance_orig.startState()}")

    all_results_data_collected_orig = []

    algorithms_to_execute_orig = []

    if args.inference == "all":
        print(f"Selected Inference (mechanics.py): ALL. Running all algorithms sequentially for {problem_name_str_orig}.")
        algorithms_to_execute_orig = [
            ("Dynamic Programming", dynamicProgramming, None),
            ("Uniform Cost Search", uniformCostSearch_with_negative_costs, None),
            ("A* Search", a_star_search, analytical_action_to_go_heuristic)
        ]
    elif args.inference == "dp":
        print(f"Selected Inference (mechanics.py): Dynamic Programming for {problem_name_str_orig}")
        algorithms_to_execute_orig = [("Dynamic Programming", dynamicProgramming, None)]
    elif args.inference == "ucs":
        print(f"Selected Inference (mechanics.py): Uniform Cost Search for {problem_name_str_orig}")
        algorithms_to_execute_orig = [("Uniform Cost Search", uniformCostSearch_with_negative_costs, None)]
    elif args.inference == "astar":
        print(f"Selected Inference (mechanics.py): A* Search for {problem_name_str_orig}")
        algorithms_to_execute_orig = [("A* Search", a_star_search, analytical_action_to_go_heuristic)]

    for current_inference_name, search_function, heuristic_func in algorithms_to_execute_orig:
        print(f"\n--- Running (mechanics.py): {current_inference_name} ---")
        
        pastCost = float('inf')
        history = []
        runtime = 0
        memory_usage_bytes = 0

        if args.inference == "all":
            tracemalloc.start() 
            start_time = time.perf_counter()

        if heuristic_func:
            print(f"Using heuristic: {heuristic_func.__name__}")
            pastCost, history = search_function(problem_instance_orig, heuristic_func)
        else: # Pass problem_instance_orig
            pastCost, history = search_function(problem_instance_orig)

        if args.inference == "all":
            end_time = time.perf_counter()
            _, peak_mem = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            runtime = end_time - start_time
            memory_usage_bytes = peak_mem
        
        if history:
            printSolution((pastCost, history))
            calculate_and_print_rmse_against_analytical(history, problem_instance_orig, problem_name_str_orig, current_inference_name)
            if args.inference == "all":
                print(f"Runtime (mechanics.py): {runtime:.4f} seconds")
                print(f"Peak Memory Usage (mechanics.py): {memory_usage_bytes / 1024:.2f} KiB ({memory_usage_bytes} bytes)")
            
            if args.inference == "all":
                all_results_data_collected_orig.append({
                    "history": history,
                    "problem_instance": problem_instance_orig,
                    "problem_name": problem_name_str_orig,
                    "inference_name": current_inference_name,
                    "runtime": runtime,
                    "memory_usage": memory_usage_bytes
                })
            else:
                plotSolution(history, problem_instance_orig, problem_name_str_orig, current_inference_name)
        else:
            print(f"No solution found with {current_inference_name} (mechanics.py). Total cost: {pastCost}")

    if args.inference == "all" and all_results_data_collected_orig:
        print("\n--- Generating combined plot for all successful algorithms (mechanics.py) ---")
        plot_all_solutions(all_results_data_collected_orig)
        print("\n--- Generating performance metrics plot (mechanics.py) ---")
        plot_performance_metrics(all_results_data_collected_orig)

    elif args.inference == "all" and not all_results_data_collected_orig:
        print("\n--- No successful solutions to plot in 'all' mode (mechanics.py) ---")