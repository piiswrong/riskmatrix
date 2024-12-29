import polars as pl
from scipy import stats
import numpy as np
import warnings
import matplotlib.pyplot as plt

def plot_array(title: str, array: np.ndarray, cumulative: bool):
    """
    Plot either the original array or its cumulative sum based on a boolean flag.

    Args:
    title (str): Title of the plot.
    array (numpy.ndarray): NumPy array to be plotted.
    cumulative (bool): If True, plot the cumulative sum of the array. If False, plot the original array.
    """
    if cumulative:
        # Calculate the cumulative sum of the array if the flag is True
        array_to_plot = np.cumsum(np.nan_to_num(array, nan=0.0))
        ylabel = 'Cumulative Sum'
    else:
        # Use the original array if the flag is False
        array_to_plot = array
        ylabel = 'Values'
    
    # Create the plot
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(5, 3))
    plt.plot(array_to_plot, marker='o', linestyle='-', color='blue', linewidth = 1, markersize = 1)
    # plt.title(title + ' cumulative' if cumulative else '')
    if cumulative:
        plt.title(title + ' (cumulative)')
    else:
        plt.title(title)
    plt.xlabel('Index')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()

def calculate_statistics(ic_array: np.ndarray):
    valid_ic = ic_array[~np.isnan(ic_array)]  # Filter out NaN values for statistics
    
    if len(valid_ic) == 0:
        return {}  # Return empty dictionary if no valid ICs

    mean_ic = np.mean(valid_ic)
    median_ic = np.median(valid_ic)
    std_ic = np.std(valid_ic)
    significant_count = np.sum(np.abs(valid_ic) > 0.1)  # Count ICs significantly different from 0
    significant_proportion = significant_count / len(valid_ic)
    
    # Calculate Information Ratio (IR)
    ir = mean_ic / std_ic if std_ic != 0 else float('inf')
    
    # Calculate Skewness and Kurtosis
    skewness = stats.skew(valid_ic)
    kurtosis = stats.kurtosis(valid_ic)

    # Perform t-test against the null hypothesis that mean = 0
    t_statistic, p_value = stats.ttest_1samp(valid_ic, 0)

    # Compile results into a dictionary
    results = {
        'Mean IC': mean_ic,
        'Median IC': median_ic,
        'STD IC': std_ic,
        # 'Count of Significant ICs (|IC| > 0.1)': significant_count,
        'Sign ICs Count': significant_count,
        'Significant ICs Ratio': significant_proportion,
        'IR': ir,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'T val': t_statistic,
        'P val': p_value
    }
    
    return results


def plot_histogram(data, title='Histogram', xlabel='Value', ylabel='Frequency', bins=20, color='blue'):
    """
    Plots a histogram for the given data array.
    
    Parameters:
        data (array-like): The data array for which the histogram will be plotted.
        title (str): The title of the histogram plot.
        xlabel (str): The label for the x-axis.
        ylabel (str): The label for the y-axis.
        bins (int): The number of bins in the histogram.
        color (str): The color of the histogram bars.
    """
    # Filter out NaN values for accurate plotting
    valid_data = data[~np.isnan(data)]
    
    # Create the histogram
    # plt.figure(figsize=(10, 6))
    plt.figure(figsize=(5, 3))
    plt.hist(valid_data, bins=bins, color=color, alpha=0.75)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.show()