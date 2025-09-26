import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from typing import Callable, List, Tuple, Union


class DifferentialEquationSolver:
    """
    A class to solve various types of differential equations using numerical methods.
    """

    def __init__(self):
        self.methods = {
            'euler': self.euler_method,
            'heun': self.heun_method,
            'rk4': self.runge_kutta_4,
            'adaptive_rk': self.adaptive_runge_kutta
        }

    def euler_method(self, f: Callable, y0: float, t_span: Tuple[float, float],
                     n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) using Euler's method.

        Args:
            f: Function f(t, y) defining the ODE
            y0: Initial condition y(t0)
            t_span: Tuple (t0, tf) defining the time interval
            n_steps: Number of steps to take

        Returns:
            t: Array of time points
            y: Array of solution values
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0

        for i in range(n_steps):
            y[i + 1] = y[i] + h * f(t[i], y[i])

        return t, y

    def heun_method(self, f: Callable, y0: float, t_span: Tuple[float, float],
                    n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) using Heun's method (improved Euler).

        Args:
            f: Function f(t, y) defining the ODE
            y0: Initial condition y(t0)
            t_span: Tuple (t0, tf) defining the time interval
            n_steps: Number of steps to take

        Returns:
            t: Array of time points
            y: Array of solution values
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0

        for i in range(n_steps):
            # Predictor (Euler)
            y_pred = y[i] + h * f(t[i], y[i])
            # Corrector (Heun)
            y[i + 1] = y[i] + h / 2 * (f(t[i], y[i]) + f(t[i + 1], y_pred))

        return t, y

    def runge_kutta_4(self, f: Callable, y0: float, t_span: Tuple[float, float],
                      n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) using 4th-order Runge-Kutta method.

        Args:
            f: Function f(t, y) defining the ODE
            y0: Initial condition y(t0)
            t_span: Tuple (t0, tf) defining the time interval
            n_steps: Number of steps to take

        Returns:
            t: Array of time points
            y: Array of solution values
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        y = np.zeros(n_steps + 1)
        y[0] = y0

        for i in range(n_steps):
            k1 = h * f(t[i], y[i])
            k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
            k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
            k4 = h * f(t[i] + h, y[i] + k3)

            y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        return t, y

    def adaptive_runge_kutta(self, f: Callable, y0: float, t_span: Tuple[float, float],
                             tol: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve dy/dt = f(t, y) using an adaptive Runge-Kutta method (Fehlberg).

        Args:
            f: Function f(t, y) defining the ODE
            y0: Initial condition y(t0)
            t_span: Tuple (t0, tf) defining the time interval
            tol: Tolerance for error control

        Returns:
            t: Array of time points
            y: Array of solution values
        """
        # Use SciPy's adaptive Runge-Kutta method
        sol = solve_ivp(f, t_span, [y0], method='RK45', rtol=tol, atol=tol)
        return sol.t, sol.y[0]

    def solve_ode(self, f: Callable, y0: float, t_span: Tuple[float, float],
                  method: str = 'rk4', **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a first-order ODE using the specified method.

        Args:
            f: Function f(t, y) defining the ODE
            y0: Initial condition y(t0)
            t_span: Tuple (t0, tf) defining the time interval
            method: Method to use ('euler', 'heun', 'rk4', 'adaptive_rk')
            **kwargs: Additional arguments for the method

        Returns:
            t: Array of time points
            y: Array of solution values
        """
        if method not in self.methods:
            raise ValueError(f"Method {method} not supported. Available methods: {list(self.methods.keys())}")

        return self.methods[method](f, y0, t_span, **kwargs)

    def solve_system(self, f: Callable, y0: List[float], t_span: Tuple[float, float],
                     method: str = 'rk4', n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve a system of first-order ODEs.

        Args:
            f: Function f(t, y) defining the system of ODEs
            y0: List of initial conditions
            t_span: Tuple (t0, tf) defining the time interval
            method: Method to use ('euler', 'heun', 'rk4')
            n_steps: Number of steps to take

        Returns:
            t: Array of time points
            y: Array of solution values (each column represents a variable)
        """
        t0, tf = t_span
        h = (tf - t0) / n_steps
        t = np.linspace(t0, tf, n_steps + 1)
        n_vars = len(y0)
        y = np.zeros((n_steps + 1, n_vars))
        y[0] = y0

        if method == 'euler':
            for i in range(n_steps):
                y[i + 1] = y[i] + h * f(t[i], y[i])

        elif method == 'heun':
            for i in range(n_steps):
                # Predictor (Euler)
                y_pred = y[i] + h * f(t[i], y[i])
                # Corrector (Heun)
                y[i + 1] = y[i] + h / 2 * (f(t[i], y[i]) + f(t[i + 1], y_pred))

        elif method == 'rk4':
            for i in range(n_steps):
                k1 = h * f(t[i], y[i])
                k2 = h * f(t[i] + h / 2, y[i] + k1 / 2)
                k3 = h * f(t[i] + h / 2, y[i] + k2 / 2)
                k4 = h * f(t[i] + h, y[i] + k3)

                y[i + 1] = y[i] + (k1 + 2 * k2 + 2 * k3 + k4) / 6

        else:
            raise ValueError(f"Method {method} not supported for systems.")

        return t, y

    def solve_second_order(self, f: Callable, y0: Tuple[float, float],
                           t_span: Tuple[float, float], method: str = 'rk4',
                           n_steps: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve a second-order ODE by converting to a system of first-order ODEs.

        Args:
            f: Function f(t, y, dy/dt) defining the second-order ODE
            y0: Tuple (y(t0), y'(t0)) of initial conditions
            t_span: Tuple (t0, tf) defining the time interval
            method: Method to use ('euler', 'heun', 'rk4')
            n_steps: Number of steps to take

        Returns:
            t: Array of time points
            y: Array of solution values
            dy: Array of derivative values
        """

        # Convert to system: y' = v, v' = f(t, y, v)
        def system(t, z):
            y, v = z
            return [v, f(t, y, v)]

        t, z = self.solve_system(system, list(y0), t_span, method, n_steps)
        y, dy = z[:, 0], z[:, 1]

        return t, y, dy


def plot_solution(t: np.ndarray, y: np.ndarray, title: str = "Solution",
                  xlabel: str = "t", ylabel: str = "y", label: str = None):
    """
    Plot the solution of a differential equation.

    Args:
        t: Time points
        y: Solution values
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        label: Curve label for legend
    """
    plt.figure(figsize=(10, 6))
    plt.plot(t, y, label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if label:
        plt.legend()
    plt.grid(True)
    plt.show()


def compare_methods(solver: DifferentialEquationSolver, f: Callable, y0: float,
                    t_span: Tuple[float, float], exact_solution: Callable = None):
    """
    Compare different numerical methods for solving an ODE.

    Args:
        solver: Differential equation solver instance
        f: Function defining the ODE
        y0: Initial condition
        t_span: Time interval
        exact_solution: Exact solution function (if available)
    """
    methods = ['euler', 'heun', 'rk4']
    n_steps = 100

    plt.figure(figsize=(12, 8))

    # Plot exact solution if available
    if exact_solution:
        t_exact = np.linspace(t_span[0], t_span[1], 1000)
        y_exact = exact_solution(t_exact)
        plt.plot(t_exact, y_exact, 'k-', linewidth=2, label='Exact')

    # Plot numerical solutions
    for method in methods:
        t, y = solver.solve_ode(f, y0, t_span, method, n_steps=n_steps)
        plt.plot(t, y, '--', linewidth=1.5, label=method.upper())

    plt.title('Comparison of Numerical Methods')
    plt.xlabel('t')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage and demonstration
if __name__ == "__main__":
    solver = DifferentialEquationSolver()

    print("Differential Equation Solver Demonstration")
    print("=" * 50)

    # Example 1: Simple exponential growth
    print("\n1. Exponential Growth: dy/dt = y, y(0) = 1")


    def exponential_growth(t, y):
        return y


    def exact_exponential(t):
        return np.exp(t)


    t_span = (0, 2)
    t, y = solver.solve_ode(exponential_growth, 1, t_span, method='rk4', n_steps=50)
    plot_solution(t, y, "Exponential Growth", label="Numerical")

    # Compare methods
    compare_methods(solver, exponential_growth, 1, t_span, exact_exponential)

    # Example 2: Harmonic oscillator (second-order)
    print("\n2. Harmonic Oscillator: d²y/dt² + y = 0, y(0)=0, y'(0)=1")


    def harmonic_oscillator(t, y, dy):
        return -y  # d²y/dt² = -y


    t, y, dy = solver.solve_second_order(harmonic_oscillator, (0, 1), (0, 4 * np.pi),
                                         method='rk4', n_steps=200)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(t, y, 'b-', label='Position')
    plt.title('Harmonic Oscillator')
    plt.ylabel('y(t)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, dy, 'r-', label='Velocity')
    plt.xlabel('t')
    plt.ylabel("y'(t)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example 3: Lotka-Volterra predator-prey model (system of ODEs)
    print("\n3. Lotka-Volterra Predator-Prey Model")


    def lotka_volterra(t, z, alpha=1.1, beta=0.4, gamma=0.4, delta=0.1):
        x, y = z  # x: prey, y: predator
        dxdt = alpha * x - beta * x * y
        dydt = delta * x * y - gamma * y
        return np.array([dxdt, dydt])


    # Wrap for our solver (needs to accept t and z)
    def lotka_volterra_wrapped(t, z):
        return lotka_volterra(t, z)


    t, z = solver.solve_system(lotka_volterra_wrapped, [10, 5], (0, 50),
                               method='rk4', n_steps=1000)

    plt.figure(figsize=(12, 8))
    plt.plot(t, z[:, 0], 'g-', label='Prey (x)')
    plt.plot(t, z[:, 1], 'r-', label='Predator (y)')
    plt.title('Lotka-Volterra Predator-Prey Model')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Phase portrait
    plt.figure(figsize=(10, 8))
    plt.plot(z[:, 0], z[:, 1], 'b-')
    plt.plot(z[0, 0], z[0, 1], 'go', markersize=8, label='Start')
    plt.plot(z[-1, 0], z[-1, 1], 'ro', markersize=8, label='End')
    plt.title('Phase Portrait: Predator vs Prey')
    plt.xlabel('Prey Population')
    plt.ylabel('Predator Population')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Example 4: Adaptive step size demonstration
    print("\n4. Adaptive Step Size Demonstration: dy/dt = -100y, y(0)=1")


    def stiff_ode(t, y):
        return -100 * y


    t_adaptive, y_adaptive = solver.solve_ode(stiff_ode, 1, (0, 1),
                                              method='adaptive_rk', tol=1e-6)
    t_fixed, y_fixed = solver.solve_ode(stiff_ode, 1, (0, 1),
                                        method='rk4', n_steps=100)

    plt.figure(figsize=(12, 8))
    plt.semilogy(t_adaptive, y_adaptive, 'b-', label='Adaptive RK (variable step)')
    plt.semilogy(t_fixed, y_fixed, 'ro', label='RK4 (fixed step)')
    plt.title('Adaptive vs Fixed Step Size for Stiff ODE')
    plt.xlabel('t')
    plt.ylabel('y(t) (log scale)')
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"Adaptive method used {len(t_adaptive)} points")
    print(f"Fixed step method used {len(t_fixed)} points")