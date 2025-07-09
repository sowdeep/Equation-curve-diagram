import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import pandas as pd
import os # Import os module for path manipulation

def get_user_input():
    """
    Gets X and Y values from a CSV or Excel file specified by the user.
    The file is expected to have 'X' and 'Y' columns.
    """
    while True:
        file_name = input("Enter the CSV/Excel file name (e.g., data.csv or data.xlsx): ")
        # Construct the full path assuming the file is in the same directory as the script
        # In a typical execution environment, the current working directory might be where the script is.
        # For a specific path like C:\Users\aaa\Desktop\equation curve, you could hardcode it or make it configurable.
        # For now, assuming the script is run from or near that directory.
        # If the script is NOT in C:\Users\aaa\Desktop\equation curve, you'd need to provide the full path here.
        # For this example, we'll assume the script is in the same directory as the data.
        file_path = os.path.join(os.getcwd(), file_name)

        try:
            if file_name.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
                df = pd.read_excel(file_path)
            else:
                print("Unsupported file format. Please use a .csv, .xlsx, or .xls file.")
                continue

            if 'X' not in df.columns or 'Y' not in df.columns:
                print("Error: The file must contain 'X' and 'Y' columns. Please check your file headers.")
                continue

            x_values = df['X'].to_numpy(dtype=float)
            y_values = df['Y'].to_numpy(dtype=float)

            if len(x_values) != len(y_values):
                print("Error: X and Y value counts do not match in the file. Please check your data.")
                continue
            if len(x_values) < 2:
                print("Error: The file must contain at least two data points.")
                continue

            print(f"Successfully loaded data from '{file_name}'.")
            return x_values, y_values

        except FileNotFoundError:
            print(f"Error: File '{file_name}' not found at '{file_path}'. Please ensure the file is in the correct directory and the name is spelled correctly.")
        except pd.errors.EmptyDataError:
            print(f"Error: File '{file_name}' is empty.")
        except KeyError as e:
            print(f"Error: Missing expected column in file: {e}. Please ensure 'X' and 'Y' columns exist.")
        except ValueError:
            print("Error: Data in 'X' or 'Y' columns could not be converted to numbers. Please check your file content.")
        except Exception as e:
            print(f"An unexpected error occurred while reading the file: {e}. Please try again.")

def calculate_r_squared(y_true, y_pred):
    """Calculates the R-squared value."""
    # Ensure y_true and y_pred have the same length and are not empty
    if len(y_true) == 0 or len(y_true) != len(y_pred):
        return np.nan # Return Not a Number if comparison is not possible
    return r2_score(y_true, y_pred)

# --- Define functions for curve fitting ---

def func_exponential(x, a, b):
    """Exponential function: y = a * exp(b * x)"""
    return a * np.exp(b * x)

def func_logarithmic(x, a, b):
    """Logarithmic function: y = a * ln(x) + b"""
    # Ensure x is positive for logarithm
    # Add a small epsilon to x to avoid log(0) for plotting if x_sorted contains 0
    return a * np.log(x + 1e-9) + b # Added epsilon for robustness

def func_power(x, a, b):
    """Power function: y = a * x^b"""
    # Ensure x is positive for power function
    return a * (x**b)

def func_inverse(x, a):
    """Inverse (Reciprocal) function: y = a / x"""
    # Avoid division by zero
    return a / (x + 1e-9) # Added epsilon for robustness

def func_inverse_square(x, a):
    """Inverse Square function: y = a / x^2"""
    # Avoid division by zero
    return a / ((x + 1e-9)**2) # Added epsilon for robustness

def func_sine(x, a, b, c):
    """Sine function: y = a * sin(b * x + c)"""
    return a * np.sin(b * x + c)

def func_cosine(x, a, b, c):
    """Cosine function: y = a * cos(b * x + c)"""
    return a * np.cos(b * x + c)

def func_tangent(x, a, b, c):
    """Tangent function: y = a * tan(b * x + c)"""
    return a * np.tan(b * x + c)

def func_sigmoid(x, L, k, x0):
    """Sigmoid (Logistic) function: y = L / (1 + e^(-k * (x - x0)))"""
    return L / (1 + np.exp(-k * (x - x0)))

def func_gaussian(x, a, b, c):
    """Gaussian (Normal Distribution) function: y = a * e^(- (x - b)^2 / (2 * c^2))"""
    return a * np.exp(-(x - b)**2 / (2 * c**2))

def func_hyperbolic_sine(x, a, b):
    """Hyperbolic Sine function: y = a * sinh(b * x)"""
    return a * np.sinh(b * x)

def func_hyperbolic_cosine(x, a, b):
    """Hyperbolic Cosine function: y = a * cosh(b * x)"""
    return a * np.cosh(b * x)

def func_damped_oscillation(x, a, b, c, d):
    """Damped Oscillation function: y = a * e^(-b * x) * sin(c * x + d)"""
    return a * np.exp(-b * x) * np.sin(c * x + d)

# A simple piecewise linear function example (user would define the split point 'd')
# For a general solution, this would need to be more dynamic based on user input.
def func_piecewise_linear(x, a1, b1, a2, b2, d):
    """
    Example of a piecewise linear function:
    y = a1 * x + b1 if x < d
    y = a2 * x + b2 if x >= d
    """
    return np.piecewise(x, [x < d, x >= d], [lambda x: a1 * x + b1, lambda x: a2 * x + b2])


def plot_and_save_trendlines(x_data, y_data):
    """
    Generates a separate scatter plot for each trendline,
    saves the plot as an image, and records the equation and R-squared value.
    """
    # Sort data for smooth trendline plotting
    sort_indices = np.argsort(x_data)
    x_sorted = x_data[sort_indices]
    y_sorted = y_data[sort_indices]

    results = [] # To store equation and R-squared for all trendlines

    output_dir = os.path.join(os.getcwd(), "trendline_plots")
    os.makedirs(output_dir, exist_ok=True) # Create a directory for plots if it doesn't exist

    print(f"\nSaving plots and results to: {output_dir}")

    # Helper function to plot and save
    def plot_and_save(x_plot, y_plot, trendline_label, file_prefix, equation, r2):
        plt.figure(figsize=(10, 7))
        plt.scatter(x_data, y_data, label='Original Data', color='blue', s=50, zorder=5)
        plt.plot(x_plot, y_plot, label=trendline_label, linestyle='--', color='red')
        plt.title(f'Scatter Plot with {file_prefix} Trendline\nEquation: {equation} ($R^2={r2:.3f}$)')
        plt.xlabel('X Values')
        plt.ylabel('Y Values')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{file_prefix}_trendline.png'))
        plt.close() # Close the figure to free memory

    # --- Linear Trendline ---
    trend_name = "Linear"
    try:
        coeffs_linear = np.polyfit(x_data, y_data, 1)
        y_pred_linear_plot = np.poly1d(coeffs_linear)(x_sorted)
        y_pred_linear_r2 = np.poly1d(coeffs_linear)(x_data)
        r2_linear = calculate_r_squared(y_data, y_pred_linear_r2)
        equation_linear = f'y = {coeffs_linear[0]:.2f}x + {coeffs_linear[1]:.2f}'
        plot_and_save(x_sorted, y_pred_linear_plot, f'{trend_name} Trendline', 'linear', equation_linear, r2_linear)
        results.append({'Trendline Type': trend_name, 'Equation': equation_linear, 'R-squared': r2_linear})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A', 'R-squared': np.nan})


    # --- Exponential Trendline (y = a * exp(b * x)) ---
    trend_name = "Exponential"
    try:
        p0_exp = [np.mean(y_data), 0.1]
        valid_indices = y_data > 0
        if np.any(valid_indices):
            params_exp, _ = curve_fit(func_exponential, x_data[valid_indices], y_data[valid_indices], p0=p0_exp, maxfev=5000)
            y_pred_exp_plot = func_exponential(x_sorted, *params_exp)
            y_pred_exp_r2 = func_exponential(x_data[valid_indices], *params_exp)
            r2_exp = calculate_r_squared(y_data[valid_indices], y_pred_exp_r2)
            equation_exp = f'y = {params_exp[0]:.2f} * exp({params_exp[1]:.2f}x)'
            plot_and_save(x_sorted, y_pred_exp_plot, f'{trend_name} Trendline', 'exponential', equation_exp, r2_exp)
            results.append({'Trendline Type': trend_name, 'Equation': equation_exp, 'R-squared': r2_exp})
        else:
            print(f"Skipping {trend_name} trendline: No positive Y values for fitting.")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A (No positive Y values)', 'R-squared': np.nan})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})


    # --- Logarithmic Trendline (y = a * ln(x) + b) ---
    trend_name = "Logarithmic"
    try:
        valid_indices = x_data > 0
        if np.any(valid_indices):
            params_log, _ = curve_fit(func_logarithmic, x_data[valid_indices], y_data[valid_indices], maxfev=5000)
            # Ensure x_sorted for plotting also avoids zero
            x_sorted_log = x_sorted[x_sorted > 0]
            y_pred_log_plot = func_logarithmic(x_sorted_log, *params_log)
            y_pred_log_r2 = func_logarithmic(x_data[valid_indices], *params_log)
            r2_log = calculate_r_squared(y_data[valid_indices], y_pred_log_r2)
            equation_log = f'y = {params_log[0]:.2f} * ln(x) + {params_log[1]:.2f}'
            plot_and_save(x_sorted_log, y_pred_log_plot, f'{trend_name} Trendline', 'logarithmic', equation_log, r2_log)
            results.append({'Trendline Type': trend_name, 'Equation': equation_log, 'R-squared': r2_log})
        else:
            print(f"Skipping {trend_name} trendline: No positive X values for fitting.")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A (No positive X values)', 'R-squared': np.nan})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})


    # --- Polynomial Trendlines (up to Order 6) ---
    for order in range(2, 7):
        trend_name = f"Polynomial Order {order}"
        try:
            coeffs_poly = np.polyfit(x_data, y_data, order)
            y_pred_poly_plot = np.poly1d(coeffs_poly)(x_sorted)
            y_pred_poly_r2 = np.poly1d(coeffs_poly)(x_data)
            r2_poly = calculate_r_squared(y_data, y_pred_poly_r2)
            equation_poly = 'y = ' + ' + '.join([f'{c:.2f}x^{order-i}' if i < order else f'{c:.2f}' for i, c in enumerate(coeffs_poly)])
            plot_and_save(x_sorted, y_pred_poly_plot, f'{trend_name} Trendline', f'polynomial_order_{order}', equation_poly, r2_poly)
            results.append({'Trendline Type': trend_name, 'Equation': equation_poly, 'R-squared': r2_poly})
        except Exception as e:
            print(f"Could not fit {trend_name} trendline: {e}")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A', 'R-squared': np.nan})


    # --- Power Trendline (y = a * x^b) ---
    trend_name = "Power"
    try:
        valid_indices = (x_data > 0) & (y_data > 0)
        if np.any(valid_indices):
            p0_power = [np.mean(y_data), 1] # Initial guess: a=mean(y), b=1
            params_power, _ = curve_fit(func_power, x_data[valid_indices], y_data[valid_indices], p0=p0_power, maxfev=5000)
            x_sorted_power = x_sorted[x_sorted > 0]
            y_pred_power_plot = func_power(x_sorted_power, *params_power)
            y_pred_power_r2 = func_power(x_data[valid_indices], *params_power)
            r2_power = calculate_r_squared(y_data[valid_indices], y_pred_power_r2)
            equation_power = f'y = {params_power[0]:.2f} * x^{params_power[1]:.2f}'
            plot_and_save(x_sorted_power, y_pred_power_plot, f'{trend_name} Trendline', 'power', equation_power, r2_power)
            results.append({'Trendline Type': trend_name, 'Equation': equation_power, 'R-squared': r2_power})
        else:
            print(f"Skipping {trend_name} trendline: Requires positive X and Y values for fitting.")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Requires positive X and Y)', 'R-squared': np.nan})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Moving Average ---
    trend_name = "Moving Average"
    window_size = min(3, len(y_data)) # Ensure window size is not larger than data points
    if window_size >= 2: # Moving average needs at least 2 points
        df = pd.DataFrame({'x': x_data, 'y': y_data})
        df_sorted = df.sort_values(by='x').reset_index(drop=True)
        y_ma = df_sorted['y'].rolling(window=window_size, min_periods=1, center=True).mean()
        y_ma_full = pd.Series(np.nan, index=x_data.argsort())
        y_ma_full[df_sorted.index] = y_ma.values
        y_ma_full = y_ma_full.loc[np.argsort(x_data)]
        valid_ma_indices = ~np.isnan(y_ma_full)
        if np.sum(valid_ma_indices) > 1:
            r2_ma = calculate_r_squared(y_data[valid_ma_indices], y_ma_full[valid_ma_indices])
            equation_ma = f'Window Size: {window_size}'
            plot_and_save(df_sorted['x'], y_ma, f'{trend_name} Trendline', 'moving_average', equation_ma, r2_ma)
            results.append({'Trendline Type': trend_name, 'Equation': equation_ma, 'R-squared': r2_ma})
        else:
            print(f"Skipping {trend_name} R-squared: Not enough valid points after calculating moving average.")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Not enough valid points)', 'R-squared': np.nan})
    else:
        print(f"Skipping {trend_name}: Not enough data points for specified window size.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Not enough data points)', 'R-squared': np.nan})

    # --- Inverse (Reciprocal) Trendline (y = a / x) ---
    trend_name = "Inverse"
    try:
        valid_indices = x_data != 0
        if np.any(valid_indices):
            params_inv, _ = curve_fit(func_inverse, x_data[valid_indices], y_data[valid_indices], maxfev=5000)
            x_sorted_inv = x_sorted[x_sorted != 0]
            y_pred_inv_plot = func_inverse(x_sorted_inv, *params_inv)
            y_pred_inv_r2 = func_inverse(x_data[valid_indices], *params_inv)
            r2_inv = calculate_r_squared(y_data[valid_indices], y_pred_inv_r2)
            equation_inv = f'y = {params_inv[0]:.2f} / x'
            plot_and_save(x_sorted_inv, y_pred_inv_plot, f'{trend_name} Trendline', 'inverse', equation_inv, r2_inv)
            results.append({'Trendline Type': trend_name, 'Equation': equation_inv, 'R-squared': r2_inv})
        else:
            print(f"Skipping {trend_name} trendline: No non-zero X values for fitting.")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A (No non-zero X values)', 'R-squared': np.nan})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Inverse Square Trendline (y = a / x^2) ---
    trend_name = "Inverse Square"
    try:
        valid_indices = x_data != 0
        if np.any(valid_indices):
            params_inv_sq, _ = curve_fit(func_inverse_square, x_data[valid_indices], y_data[valid_indices], maxfev=5000)
            x_sorted_inv_sq = x_sorted[x_sorted != 0]
            y_pred_inv_sq_plot = func_inverse_square(x_sorted_inv_sq, *params_inv_sq)
            y_pred_inv_sq_r2 = func_inverse_square(x_data[valid_indices], *params_inv_sq)
            r2_inv_sq = calculate_r_squared(y_data[valid_indices], y_pred_inv_sq_r2)
            equation_inv_sq = f'y = {params_inv_sq[0]:.2f} / x^2'
            plot_and_save(x_sorted_inv_sq, y_pred_inv_sq_plot, f'{trend_name} Trendline', 'inverse_square', equation_inv_sq, r2_inv_sq)
            results.append({'Trendline Type': trend_name, 'Equation': equation_inv_sq, 'R-squared': r2_inv_sq})
        else:
            print(f"Skipping {trend_name} trendline: No non-zero X values for fitting.")
            results.append({'Trendline Type': trend_name, 'Equation': 'N/A (No non-zero X values)', 'R-squared': np.nan})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Sine Function (y = a * sin(b * x + c)) ---
    trend_name = "Sine"
    try:
        amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2 if len(y_data) > 1 else 1
        p0_sine = [amplitude_guess, 1, 0]
        params_sine, _ = curve_fit(func_sine, x_data, y_data, p0=p0_sine, maxfev=5000)
        y_pred_sine_plot = func_sine(x_sorted, *params_sine)
        y_pred_sine_r2 = func_sine(x_data, *params_sine)
        r2_sine = calculate_r_squared(y_data, y_pred_sine_r2)
        equation_sine = f'y = {params_sine[0]:.2f} * sin({params_sine[1]:.2f}x + {params_sine[2]:.2f})'
        plot_and_save(x_sorted, y_pred_sine_plot, f'{trend_name} Trendline', 'sine', equation_sine, r2_sine)
        results.append({'Trendline Type': trend_name, 'Equation': equation_sine, 'R-squared': r2_sine})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found. Try different initial guesses.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Cosine Function (y = a * cos(b * x + c)) ---
    trend_name = "Cosine"
    try:
        amplitude_guess = (np.max(y_data) - np.min(y_data)) / 2 if len(y_data) > 1 else 1
        p0_cosine = [amplitude_guess, 1, 0]
        params_cosine, _ = curve_fit(func_cosine, x_data, y_data, p0=p0_cosine, maxfev=5000)
        y_pred_cosine_plot = func_cosine(x_sorted, *params_cosine)
        y_pred_cosine_r2 = func_cosine(x_data, *params_cosine)
        r2_cosine = calculate_r_squared(y_data, y_pred_cosine_r2)
        equation_cosine = f'y = {params_cosine[0]:.2f} * cos({params_cosine[1]:.2f}x + {params_cosine[2]:.2f})'
        plot_and_save(x_sorted, y_pred_cosine_plot, f'{trend_name} Trendline', 'cosine', equation_cosine, r2_cosine)
        results.append({'Trendline Type': trend_name, 'Equation': equation_cosine, 'R-squared': r2_cosine})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found. Try different initial guesses.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Tangent Function (y = a * tan(b * x + c)) ---
    trend_name = "Tangent"
    try:
        p0_tan = [1, 1, 0]
        params_tan, _ = curve_fit(func_tangent, x_data, y_data, p0=p0_tan, maxfev=5000)
        y_pred_tan_plot = func_tangent(x_sorted, *params_tan)
        y_pred_tan_r2 = func_tangent(x_data, *params_tan)
        r2_tan = calculate_r_squared(y_data, y_pred_tan_r2)
        equation_tan = f'y = {params_tan[0]:.2f} * tan({params_tan[1]:.2f}x + {params_tan[2]:.2f})'
        plot_and_save(x_sorted, y_pred_tan_plot, f'{trend_name} Trendline', 'tangent', equation_tan, r2_tan)
        results.append({'Trendline Type': trend_name, 'Equation': equation_tan, 'R-squared': r2_tan})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found. This function is sensitive to initial guesses and data range.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Sigmoid (Logistic Curve): y = L / (1 + e^(-k * (x - x0))) ---
    trend_name = "Sigmoid"
    try:
        L_guess = np.max(y_data) * 1.1
        k_guess = 1.0
        x0_guess = np.median(x_data)
        p0_sigmoid = [L_guess, k_guess, x0_guess]
        params_sigmoid, _ = curve_fit(func_sigmoid, x_data, y_data, p0=p0_sigmoid, maxfev=5000)
        y_pred_sigmoid_plot = func_sigmoid(x_sorted, *params_sigmoid)
        y_pred_sigmoid_r2 = func_sigmoid(x_data, *params_sigmoid)
        r2_sigmoid = calculate_r_squared(y_data, y_pred_sigmoid_r2)
        equation_sigmoid = f'y = {params_sigmoid[0]:.2f} / (1 + exp(-{params_sigmoid[1]:.2f} * (x - {params_sigmoid[2]:.2f})))'
        plot_and_save(x_sorted, y_pred_sigmoid_plot, f'{trend_name} Trendline', 'sigmoid', equation_sigmoid, r2_sigmoid)
        results.append({'Trendline Type': trend_name, 'Equation': equation_sigmoid, 'R-squared': r2_sigmoid})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found. Try adjusting initial guesses.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Gaussian (Normal Distribution Curve): y = a * e^(- (x - b)^2 / (2 * c^2)) ---
    trend_name = "Gaussian"
    try:
        a_guess = np.max(y_data)
        b_guess = x_data[np.argmax(y_data)]
        c_guess = (np.max(x_data) - np.min(x_data)) / 4
        p0_gaussian = [a_guess, b_guess, c_guess]
        params_gaussian, _ = curve_fit(func_gaussian, x_data, y_data, p0=p0_gaussian, maxfev=5000)
        y_pred_gaussian_plot = func_gaussian(x_sorted, *params_gaussian)
        y_pred_gaussian_r2 = func_gaussian(x_data, *params_gaussian)
        r2_gaussian = calculate_r_squared(y_data, y_pred_gaussian_r2)
        equation_gaussian = f'y = {params_gaussian[0]:.2f} * exp(- (x - {params_gaussian[1]:.2f})^2 / (2 * {params_gaussian[2]:.2f}^2))'
        plot_and_save(x_sorted, y_pred_gaussian_plot, f'{trend_name} Trendline', 'gaussian', equation_gaussian, r2_gaussian)
        results.append({'Trendline Type': trend_name, 'Equation': equation_gaussian, 'R-squared': r2_gaussian})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found. Try adjusting initial guesses.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Hyperbolic Sine: y = a * sinh(b * x) ---
    trend_name = "Hyperbolic Sine"
    try:
        p0_sinh = [1, 0.1]
        params_sinh, _ = curve_fit(func_hyperbolic_sine, x_data, y_data, p0=p0_sinh, maxfev=5000)
        y_pred_sinh_plot = func_hyperbolic_sine(x_sorted, *params_sinh)
        y_pred_sinh_r2 = func_hyperbolic_sine(x_data, *params_sinh)
        r2_sinh = calculate_r_squared(y_data, y_pred_sinh_r2)
        equation_sinh = f'y = {params_sinh[0]:.2f} * sinh({params_sinh[1]:.2f}x)'
        plot_and_save(x_sorted, y_pred_sinh_plot, f'{trend_name} Trendline', 'hyperbolic_sine', equation_sinh, r2_sinh)
        results.append({'Trendline Type': trend_name, 'Equation': equation_sinh, 'R-squared': r2_sinh})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Hyperbolic Cosine: y = a * cosh(b * x) ---
    trend_name = "Hyperbolic Cosine"
    try:
        p0_cosh = [1, 0.1]
        params_cosh, _ = curve_fit(func_hyperbolic_cosine, x_data, y_data, p0=p0_cosh, maxfev=5000)
        y_pred_cosh_plot = func_hyperbolic_cosine(x_sorted, *params_cosh)
        y_pred_cosh_r2 = func_hyperbolic_cosine(x_data, *params_cosh)
        r2_cosh = calculate_r_squared(y_data, y_pred_cosh_r2)
        equation_cosh = f'y = {params_cosh[0]:.2f} * cosh({params_cosh[1]:.2f}x)'
        plot_and_save(x_sorted, y_pred_cosh_plot, f'{trend_name} Trendline', 'hyperbolic_cosine', equation_cosh, r2_cosh)
        results.append({'Trendline Type': trend_name, 'Equation': equation_cosh, 'R-squared': r2_cosh})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})

    # --- Piecewise (Example: two linear segments) ---
    # As discussed, automatically fitting a general piecewise function is complex.
    # This section remains a conceptual placeholder.
    trend_name = "Piecewise (Conceptual)"
    print(f"\nNote: {trend_name} fitting is highly dependent on the definition of the piecewise points.")
    print("A general automatic fit for an arbitrary piecewise function is complex.")
    print("Consider splitting your data and fitting separate models to each segment if you have known split points.")
    results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Complex/Manual Fit)', 'R-squared': np.nan})


    # --- Damped Oscillation: y = a * e^(-b * x) * sin(c * x + d) ---
    trend_name = "Damped Oscillation"
    try:
        a_guess = (np.max(y_data) - np.min(y_data)) / 2
        b_guess = 0.1
        c_guess = 1.0
        d_guess = 0.0
        p0_damped = [a_guess, b_guess, c_guess, d_guess]
        params_damped, _ = curve_fit(func_damped_oscillation, x_data, y_data, p0=p0_damped, maxfev=5000)
        y_pred_damped_plot = func_damped_oscillation(x_sorted, *params_damped)
        y_pred_damped_r2 = func_damped_oscillation(x_data, *params_damped)
        r2_damped = calculate_r_squared(y_data, y_pred_damped_r2)
        equation_damped = f'y = {params_damped[0]:.2f} * exp(-{params_damped[1]:.2f}x) * sin({params_damped[2]:.2f}x + {params_damped[3]:.2f})'
        plot_and_save(x_sorted, y_pred_damped_plot, f'{trend_name} Trendline', 'damped_oscillation', equation_damped, r2_damped)
        results.append({'Trendline Type': trend_name, 'Equation': equation_damped, 'R-squared': r2_damped})
    except RuntimeError:
        print(f"Could not fit {trend_name} trendline: Optimal parameters not found. Try different initial guesses.")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Fit failed)', 'R-squared': np.nan})
    except Exception as e:
        print(f"Could not fit {trend_name} trendline: {e}")
        results.append({'Trendline Type': trend_name, 'Equation': 'N/A (Error)', 'R-squared': np.nan})


    # Save all results to a CSV file
    results_df = pd.DataFrame(results)
    results_file_path = os.path.join(output_dir, "trendline_results.csv")
    results_df.to_csv(results_file_path, index=False)
    print(f"\nAll trendline equations and R-squared values saved to: {results_file_path}")

if __name__ == "__main__":
    x_values, y_values = get_user_input()
    # Call the new function to plot and save individual trendlines
    plot_and_save_trendlines(x_values, y_values)
