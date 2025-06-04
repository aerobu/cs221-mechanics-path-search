# File name: mechanicsAndDP.py
# Author: Ajith Kosireddy
# email: ajithkosireddy9@gmail.com
# Description: Models a mechanical system as a search problem and uses dynamic programming
#              to solve the problem. Three mechanical systems are modeled:
#                  1. free particle
#                  2. free fall (particle in a uniform gravitational field)
#                  3. simple harmonic motion
#              The only performance measures used in this code are runtime and visual
#              inspection of plots. The free particle and free fall systems give the expected
#              behavior. The harmonic oscillator runs into a problem with the maximum speed limit
#              (explained below).

from dataclasses import dataclass
import matplotlib.pyplot as plt
import time

###### Modeling #####

@dataclass(frozen=True, eq=True)
class State:
    t     : float
    q     : float
    qdot  : float

# This is a base class with no implementation of the function |calcDeltaS|. The systems below
# implement this function based on the particular physical system being simulated.
class MechanicalSystem:
    def __init__(self, q0: int = 5, qf: int = 95, t0: int = 0, tf: int = 18, dt: int = 1, max_speed: int = 10, qmin: int = 0, qmax: int = 3000, m: float = 1):
        self.m: float = m
        self.q0: int = q0
        self.qf: int = qf
        self.t0: int = t0
        self.tf: int = tf
        self.dt: int = dt
        self.qmin: int = qmin
        self.qmax: int = qmax
        self.maxSpeed: int = max_speed
        self.possibleVelocities: List[int] = list(range(-self.maxSpeed, self.maxSpeed + 1))
    def startState(self):
        t = 0
        q = self.q0
        qdot = None
        return State(t, q, qdot)
    def isEnd(self, state):
        return state.t == self.tf and state.q == self.qf
    def succAndCost(self, state):
        t_current = state.t
        q = state.q
        if t_current >= self.tf:
            return []
        t_next = t_current + self.dt
        retList = []
        for v in self.possibleVelocities:
            cost = self.calcDeltaS(q, v)
            q_next = round(q + v * self.dt)
            if q_next > self.qmax or q_next < self.qmin:
                continue
            nextState = State(t_next, q_next, v)
            retList.append((v, nextState, cost)) # (action, nextState, cost)
        return retList
    def calcDeltaS(self, q, qdot):
        pass

class FreeParticle(MechanicalSystem):
    def calcDeltaS(self, q, qdot):
        # (Lagrangian) * (time step)
        return 0.5 * self.m * (qdot**2) * self.dt

class FreeFall(MechanicalSystem):
    def calcDeltaS(self, q, qdot):
        self.g = 2
        return (0.5 * self.m * (qdot**2) - self.m * self.g * q) * self.dt # (Lagrangian) * (time step)


###### Harmonic Oscillator ######
# High values of the total energy are problematic -- they conflict with the |self.maxSpeed| limit!
# If you run this problem, you will likely see the velocity of the system reaching the maximim or
# minimum possible value. Instead of seeing the expected sinusoidal behavior, you may see something
# like a triangle function for the output path.
#     If we increase |self.maxSpeed| to allow the system to reach velocities of higher magnitude, the
# action space of the search problem increases. Therefore, runtime will increase. There is a tradeoff
# between the speed limit and size of the action space. In other words, there is a tradeoff between
# faithfully simulating the physical system and computational complexity.
# TODO: Clever adustment of physical parameters and/or simulation parameters may reproduce the expected sinusoidal motion.
class HarmonicOscillator(MechanicalSystem):
    def calcDeltaS(self, q, qdot):
        self.omega = 1
        return (0.5 * self.m * (qdot**2) - (0.5) * self.m * (self.omega)**2 * q**2) * self.dt
    

###### Inference ######

# Base class for solving mechanical problems
class MechanicalSystemSolver:
    def __init__(self, problem):
      pass
    def solve(self):
      pass

def dynamicProgramming(problem):
    cache = {}  # state => futureCost(state), action, newState, cost
    def futureCost(state):
        # Base case
        if problem.isEnd(state):
            return 0
        if state in cache:
            return cache[state][0]
        nextStates = [] # list of tuples of form (cost + futureCost(state), action, newState, cost)
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

    # Recover history
    history = []
    history.append((None, state, None)) # (vel, (time, pos, vel), cost)
    while not problem.isEnd(state):
        _, action, newState, cost = cache[state]
        history.append((action, newState, cost))
        state = newState

    return (totalCost, history)

# Only runtime performance measure
class Evaluator:
    def run(self, problem, inferenceAlgorithm):
        start = time.perf_counter()
        totalCost, history = inferenceAlgorithm(problem)
        end = time.perf_counter()
        print(f"Execution time: {end - start:.6f} seconds\n")
        return totalCost, history

# PRINTING AND PLOTTING
def printSolution(solution):
    totalCost, history = solution
    print('totalCost:', totalCost)
    for item in history:
        print(item)

def plotSolution(history, ylim1=None, ylim2=None, figs=None):
    t = []
    x = []
    v = []
    for item in history:
        t.append(item[1].t)
        x.append(item[1].q)
        if history[0] is None:
            v.append(v[-1])
        else:
            v.append(item[0])
    plt.figure(1)
    plt.plot(t,x,'.')
    plt.title("Position as a Function of Time")
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("x")
    if ylim1 != None and ylim2 != None:
        plt.ylim(ylim1, ylim2)
    plt.figure(2)
    plt.plot(t,v,'.')
    plt.title("Velocity as a Function of Time")
    plt.grid()
    plt.xlabel("t")
    plt.ylabel("v")


# MAIN
problem = HarmonicOscillator(q0=0, qf=0, max_speed=20, tf=18)
evaluator = Evaluator()
totalCost, history = evaluator.run(problem, dynamicProgramming)
plotSolution(history)
plt.show()
