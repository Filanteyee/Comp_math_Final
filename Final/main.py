import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, newton
from scipy.integrate import fixed_quad
from scipy.linalg import lu
from scipy.interpolate import lagrange
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Function to parse coefficients, handling 'e' and '2.71' as Euler's number
def parse_coefficients(coeff_input):
    try:
        coefficients = [float(c.strip()) if c.strip() not in ['e', '2.71'] else np.e for c in coeff_input.split(",")]
        return coefficients
    except ValueError as ve:
        raise ValueError(f"Invalid coefficient format: {coeff_input}. Error: {ve}")


# Function to create the Tkinter window
def create_window():
    window = tk.Tk()
    window.title("Computational Mathematics Interface")
    window.geometry("800x600")

    # Functionality for dynamic method selection
    methods = ["Graphical Method", "Root Finding (Bisection)", "Root Finding (Newton)", "Gauss-Seidel",
               "LU Factorization", "Polynomial Curve Fitting", "Lagrange Interpolation", "Euler's Method",
               "Boole's Rule"]
    method_label = tk.Label(window, text="Select Method:")
    method_label.pack()
    method_combobox = ttk.Combobox(window, values=methods, state="readonly")
    method_combobox.set(methods[0])  # Default method
    method_combobox.pack()

    # Input fields for coefficients, equations, etc.
    input_label = tk.Label(window, text="Enter Parameters:")
    input_label.pack()

    # Coefficients and matrix input section
    coefficient_label = tk.Label(window, text="Enter coefficients (comma-separated, use '2.71' for Euler's number):")
    coefficient_label.pack()
    coefficient_entry = tk.Entry(window)
    coefficient_entry.pack()

    # Interval input field (hidden initially)
    interval_label = tk.Label(window, text="Enter interval (comma-separated, e.g., '-2,2'): ")
    interval_label.pack()
    interval_entry = tk.Entry(window)
    interval_entry.pack()

    # Matrix input section (hidden initially)
    matrix_label = tk.Label(window, text="Enter matrix (comma-separated values for rows, e.g., '1,2,3;4,5,6'): ")
    matrix_label.pack()
    matrix_entry = tk.Entry(window)
    matrix_entry.pack()

    # Vertical vector for Task 3 (fixed values 12, 15, 10)
    vector_label = tk.Label(window, text="Enter vertical vector (comma-separated, e.g., '12,15,10'): ")
    vector_label.pack()
    vector_entry = tk.Entry(window)
    vector_entry.insert(0, "12,15,10")
    vector_entry.pack()

    # Task 5 & 6 Data Points Input
    points_label = tk.Label(window, text="Enter points (comma-separated, e.g., (0,0),(1,1),(2,8)):")
    points_label.pack()
    points_entry = tk.Entry(window)
    points_entry.pack()

    # Euler's Method Fields
    euler_initial_label = tk.Label(window, text="Enter initial value (y0):")
    euler_initial_label.pack()
    euler_initial_entry = tk.Entry(window)
    euler_initial_entry.pack()

    euler_step_label = tk.Label(window, text="Enter step size (h):")
    euler_step_label.pack()
    euler_step_entry = tk.Entry(window)
    euler_step_entry.pack()

    euler_end_label = tk.Label(window, text="Enter end value (x_end):")
    euler_end_label.pack()
    euler_end_entry = tk.Entry(window)
    euler_end_entry.pack()

    subinterval_label = tk.Label(window, text="Enter number of subintervals for Boole's rule:")
    subinterval_label.pack()
    subinterval_entry = tk.Entry(window)
    subinterval_entry.pack()

    # Result display area
    result_label = tk.Label(window, text="Results will be displayed here:")
    result_label.pack()

    # Canvas for graph
    canvas_frame = tk.Frame(window)
    canvas_frame.pack()

    # Clear canvas to avoid duplicate graphs
    def clear_graph():
        for widget in canvas_frame.winfo_children():
            widget.destroy()

    # Button for execution
    def execute_method():
        try:
            selected_method = method_combobox.get()
            coefficients = coefficient_entry.get()
            matrix_input = matrix_entry.get()
            interval_input = interval_entry.get()
            points_input = points_entry.get()
            euler_initial = euler_initial_entry.get()
            euler_step = euler_step_entry.get()
            euler_end = euler_end_entry.get()
            subintervals = subinterval_entry.get()
            vector_input = vector_entry.get()

            # Parse coefficients
            if coefficients:
                coefficients = parse_coefficients(coefficients)

            # Parse interval input
            if interval_input:
                a, b = map(float, interval_input.split(","))
                interval = (a, b)
            else:
                interval = None

            # Parse points input
            if points_input:
                points = parse_points(points_input)
            else:
                points = None

            # Parse vertical vector input
            if vector_input:
                vector = list(map(float, vector_input.split(",")))
            else:
                vector = None

            # Clear previous plots and results
            clear_graph()

            # Execute the selected method
            if selected_method == "Graphical Method":
                plot_graph(result_label, coefficients)
            elif selected_method == "Root Finding (Bisection)":
                bisection_method(result_label, interval, coefficients)
            elif selected_method == "Root Finding (Newton)":
                newton_method(result_label, interval, coefficients)
            elif selected_method == "Euler's Method":
                euler_method(result_label, euler_initial, euler_step, euler_end)
            elif selected_method == "Polynomial Curve Fitting":
                polynomial_curve_fitting(result_label, points)
            elif selected_method == "Lagrange Interpolation":
                lagrange_interpolation(result_label, points)
            elif selected_method == "Boole's Rule":
                booles_rule(result_label, subintervals)
            elif selected_method == "Gauss-Seidel":
                gauss_seidel_method(result_label, matrix_input, vector_input)
            elif selected_method == "LU Factorization":
                lu_factorization(result_label, matrix_input)

        except Exception as e:
            result_label.config(text=f"Error: {str(e)}")

    execute_button = tk.Button(window, text="Execute", command=execute_method)
    execute_button.pack()

    # Function to parse matrix input with validation
    def parse_matrix(matrix_input):
        if not matrix_input.strip():
            raise ValueError("Matrix input cannot be empty.")
        rows = matrix_input.split(";")  # Split rows based on semicolons
        matrix = []
        for row in rows:
            row = row.strip()
            if row:
                try:
                    # Ensure each row is split into a list of floats
                    matrix.append(list(map(float, row.split(","))))  # Split columns by commas
                except ValueError:
                    raise ValueError(f"Invalid number format in row: {row}")
        if not matrix:
            raise ValueError("No valid matrix rows found.")
        return np.array(matrix)  # Convert to numpy array for matrix operations

    # Function to parse points input with validation
    def parse_points(points_input):
        try:
            points = [tuple(map(float, point.strip("()").split(","))) for point in points_input.split("),(")]
            return points
        except ValueError:
            raise ValueError(f"Invalid points format: {points_input}")

    # Function to parse interval input with validation
    def parse_interval(interval_input):
        if not interval_input.strip():
            raise ValueError("Interval input cannot be empty.")
        try:
            a, b = map(float, interval_input.split(","))
            return a, b
        except ValueError:
            raise ValueError("Invalid interval format. Please enter two numbers separated by a comma.")

    # Function to plot the graph for the Graphical Method
    def plot_graph(result_label, coefficients):
        coef = coefficients
        equation = " + ".join([f"{c}x^{i}" for i, c in enumerate(reversed(coef))])

        def f(x):
            return sum(c * x ** i for i, c in enumerate(reversed(coef)))

        x = np.linspace(-2, 2, 400)
        y = f(x)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axhline(0, color='black', linewidth=1)
        ax.axvline(0, color='black', linewidth=1)
        ax.set_title(f"Graph of f(x) = {equation}")
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)

        # Embed the graph in the Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
        canvas.draw()
        canvas.get_tk_widget().pack()

    # Bisection Method Implementation (Task 1 and 2)
    def bisection_method(result_label, interval, coefficients):
        a, b = interval
        coef = coefficients

        def f(x):
            return sum(c * x ** i for i, c in enumerate(reversed(coef)))

        fa = f(a)
        fb = f(b)

        if fa * fb > 0:
            result_label.config(text=f"Error: f(a) and f(b) must have different signs for the Bisection Method.\n"
                                     f"f(a) = {fa}, f(b) = {fb}\nPlease try a different interval.")
            return

        # Track iterations
        iterations = 0
        root = None
        while (b - a) / 2 > 1e-6:
            iterations += 1
            c = (a + b) / 2
            fc = f(c)
            if fc == 0:
                break
            if f(a) * fc < 0:
                b = c
            else:
                a = c
            root = c

        result_label.config(text=f"Root found: {root} in {iterations} iterations.")
        plot_graph(result_label, coefficients)  # Add graph for the method
        return root

    # Newton's Method Implementation
    def newton_method(result_label, interval, coefficients):
        a, b = interval
        coef = coefficients

        def g(x):
            return sum(c * x ** i for i, c in enumerate(reversed(coef)))

        # Track iterations
        iterations = 0
        root = newton(g, x0=1, maxiter=100, full_output=True)
        result_label.config(text=f"Root found using Newton-Raphson: {root[0]}\nIterations: {iterations}")
        plot_graph(result_label, coefficients)  # Add graph for the method

    # Euler's Method Implementation
    def euler_method(result_label, initial_value, step_size, end_value):
        def dydx(x, y):
            return x - y

        def euler_method_solution(dydx, x0, y0, h, x_end):
            x_values = [x0]
            y_values = [y0]
            x = x0
            y = y0
            while x < x_end:
                y += h * dydx(x, y)
                x += h
                x_values.append(x)
                y_values.append(y)
            return np.array(x_values), np.array(y_values)

        x_values, y_values = euler_method_solution(dydx, 0, float(initial_value), float(step_size), float(end_value))

        result_text = "Euler's Method Steps:\n"
        for x, y in zip(x_values, y_values):
            result_text += f"x = {x}, y = {y}\n"

        result_label.config(text=result_text)

    # Polynomial Curve Fitting Implementation
    def polynomial_curve_fitting(result_label, points):
        x_data, y_data = zip(*points)
        coefficients = np.polyfit(x_data, y_data, 3)
        polynomial = np.poly1d(coefficients)

        x_fit = np.linspace(min(x_data), max(x_data), 100)
        y_fit = polynomial(x_fit)

        result_label.config(text="Polynomial Curve Fitting Complete")
        plot_graph(result_label, coefficients)  # Add graph for the method

    # Lagrange Interpolation Method Implementation (Task 6)
    def lagrange_interpolation(result_label, points):
        x_points, y_points = zip(*points)

        def lagrange_interpolation_formula(x, x_points, y_points):
            result = 0
            for i in range(len(x_points)):
                term = y_points[i]
                for j in range(len(x_points)):
                    if j != i:
                        term *= (x - x_points[j]) / (x_points[i] - x_points[j])
                result += term
            return result

        # Estimate f(7)
        estimate = lagrange_interpolation_formula(7, x_points, y_points)
        result_label.config(text=f"Estimated f(7) using Lagrange: {estimate}")

    # Boole's Rule Implementation
    def booles_rule(result_label, subintervals):
        def func(x):
            return np.exp(-x)

        # Apply Boole's Rule for integration
        integral, _ = fixed_quad(func, 1, 3, n=int(subintervals))
        exact_integral = np.exp(-1) - np.exp(-3)
        result_label.config(text=f"Integral using Boole's Rule: {integral}\nExact Integral: {exact_integral}")

    # Gauss-Seidel Method Implementation (Task 3)
    def gauss_seidel_method(result_label, matrix_input, vector_input):
        A = parse_matrix(matrix_input)
        b = np.array(list(map(float, vector_input.split(","))))  # Fixed vector parsing

        # Check for matrix and vector dimension mismatch
        if A.shape[0] != b.shape[0]:
            result_label.config(
                text=f"Error: The number of rows in the matrix ({A.shape[0]}) must match the size of the vector ({b.shape[0]}).")
            return

        solution = gauss_seidel(A, b, np.zeros(len(b)))
        result_label.config(text=f"Solution using Gauss-Seidel: {solution}")

    # LU Factorization Implementation
    def lu_factorization(result_label, matrix_input):
        A = parse_matrix(matrix_input)
        P, L, U = lu(A)

        result_label.config(text=f"L matrix: \n{L}\n\nU matrix: \n{U}")

    # Gauss-Seidel Method (Auxiliary function)
    def gauss_seidel(A, b, x0, tol=1e-6, max_iter=100):
        x = x0.copy()
        for _ in range(max_iter):
            x_new = x.copy()
            for i in range(len(b)):
                sigma = np.dot(A[i, :], x_new) - A[i, i] * x_new[i]
                x_new[i] = (b[i] - sigma) / A[i, i]
            if np.linalg.norm(x_new - x, ord=np.inf) < tol:
                return x_new
            x = x_new
        return x

    # Dynamically show/hide matrix entry field and interval entry field based on selected method
    def on_method_change(event):
        selected_method = method_combobox.get()

        # Clear the graph when the method changes
        clear_graph()

        if selected_method == "Graphical Method":
            coefficient_label.pack()
            coefficient_entry.pack()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            matrix_label.pack_forget()
            matrix_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
            subinterval_label.pack_forget()
            subinterval_entry.pack_forget()
            euler_end_label.pack_forget()
            euler_end_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()
            euler_step_label.pack_forget()
            euler_step_entry.pack_forget()
        elif selected_method in ["Root Finding (Bisection)", "Root Finding (Newton)"]:
            coefficient_label.pack()
            coefficient_entry.pack()
            interval_label.pack()
            interval_entry.pack()
            matrix_label.pack_forget()
            matrix_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()
            euler_step_label.pack_forget()
            euler_step_entry.pack_forget()
            euler_end_label.pack_forget()
            euler_end_entry.pack_forget()
            subinterval_label.pack_forget()
            subinterval_entry.pack_forget()
        elif selected_method == "Euler's Method":
            euler_initial_label.pack()
            euler_initial_entry.pack()
            euler_step_label.pack()
            euler_step_entry.pack()
            euler_end_label.pack()
            euler_end_entry.pack()
            coefficient_label.pack_forget()
            coefficient_entry.pack_forget()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            matrix_label.pack_forget()
            matrix_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
        elif selected_method == "Polynomial Curve Fitting" or selected_method == "Lagrange Interpolation":
            points_label.pack()
            points_entry.pack()
            coefficient_label.pack_forget()
            coefficient_entry.pack_forget()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()
            euler_step_label.pack_forget()
            euler_step_entry.pack_forget()
            euler_end_label.pack_forget()
            euler_end_entry.pack_forget()
            matrix_label.pack_forget()
            matrix_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()
        elif selected_method == "Boole's Rule":
            subinterval_label.pack()
            subinterval_entry.pack()
            matrix_label.pack_forget()
            matrix_entry.pack_forget()
            coefficient_label.pack_forget()
            coefficient_entry.pack_forget()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()
            euler_end_label.pack_forget()
            euler_end_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()
            euler_step_label.pack_forget()
            euler_step_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()

        elif selected_method == "Gauss-Seidel":
            matrix_label.pack()
            matrix_entry.pack()
            vector_label.pack()
            vector_entry.pack()
            coefficient_label.pack_forget()
            coefficient_entry.pack_forget()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()
            euler_step_label.pack_forget()
            euler_step_entry.pack_forget()
            euler_end_label.pack_forget()
            euler_end_entry.pack_forget()
        elif selected_method == "LU Factorization":
            matrix_label.pack()
            matrix_entry.pack()
            coefficient_label.pack_forget()
            coefficient_entry.pack_forget()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()
            euler_initial_label.pack_forget()
            euler_initial_entry.pack_forget()
            euler_step_label.pack_forget()
            euler_step_entry.pack_forget()
            euler_end_label.pack_forget()
            euler_end_entry.pack_forget()
        else:
            matrix_label.pack_forget()
            matrix_entry.pack_forget()
            interval_label.pack_forget()
            interval_entry.pack_forget()
            points_label.pack_forget()
            points_entry.pack_forget()
            vector_label.pack_forget()
            vector_entry.pack_forget()


    method_combobox.bind("<<ComboboxSelected>>", on_method_change)

    return window


# Main execution
if __name__ == "__main__":
    window = create_window()
    window.mainloop()
