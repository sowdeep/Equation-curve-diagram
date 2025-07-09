# Equation-curve-diagram
The script trendline_plotter.py takes a CSV or Excel file as input, which must contain 'X' and 'Y' columns. It then calculates and plots various trendlines, including linear, polynomial, exponential, logarithmic, and power. Other supported trendlines are moving average, inverse, inverse square, sine, cosine, tangent, sigmoid, gaussian, hyperbolic sine, hyperbolic cosine, and damped oscillation. Each generated plot, along with its equation and R-squared value, is saved as a separate PNG image in a trendline_plots directory. Additionally, a trendline_results.csv file summarizes all fitted trendlines' equations and R-squared values.
✅
To run this script, you need to install the following Python libraries: matplotlib, numpy, scipy, scikit-learn, pandas, and openpyxl (for Excel file support). You can install them using pip: pip install matplotlib numpy scipy scikit-learn pandas openpyxl.

Directory:
Output directory for plots: trendline_plots (this directory is created in the current working directory where the script is run).
Input Format:
The script prompts the user to enter the name of a CSV or Excel file.
The specified file must contain two columns, specifically named 'X' and 'Y'.
Output Format:
Individual PNG image files are generated for each fitted trendline (e.g., linear_trendline.png, exponential_trendline.png) and saved within the trendline_plots directory.
A single CSV file named trendline_results.csv is created, summarizing the type, equation, and R-squared value for each trendline that was fitted.
Command-line messages provide feedback on file loading status and the paths where output files are saved.
