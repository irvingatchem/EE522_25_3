import os
import time
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt


def save_experiment_data(
    data: Dict[str, np.ndarray],
    base_dir: str = "results",
    prefix: Optional[str] = None
) -> str:
    """
    Save multiple numpy arrays to text files in a timestamped directory.

    Args:
        data: Mapping of filename (without extension) to numpy array.
        base_dir: Base directory under which to create timestamped folder.
        prefix: Optional prefix for the folder name.

    Returns:
        Path to the directory where files are saved.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    folder_name = f"{prefix + '_'}{timestamp}" if prefix else timestamp
    dir_path = os.path.join(base_dir, folder_name)
    os.makedirs(dir_path, exist_ok=True)

    for name, arr in data.items():
        file_path = os.path.join(dir_path, f"{name}.txt")
        np.savetxt(file_path, arr, fmt="%s")
    return dir_path


def perform_paired_ttests(
    true_vals: List[List[float]],
    method_vals: Dict[str, List[List[float]]],
    depth_index: int
) -> Dict[str, Tuple[float, float]]:
    """
    Perform paired t-tests between true values and each method at a specific depth.

    Args:
        true_vals: List of trials for true values per depth.
        method_vals: Dict mapping method name to list of trials per depth.
        depth_index: Index of depth at which to compare.

    Returns:
        Mapping of method name to (t_statistic, p_value).
    """
    results: Dict[str, Tuple[float, float]] = {}
    true_array = np.array(true_vals, dtype=float)[depth_index]
    for method, vals in method_vals.items():
        arr = np.array(vals, dtype=float)[depth_index]
        if arr.shape != true_array.shape:
            continue
        t_stat, p_val = ttest_rel(arr, true_array)
        results[method] = (t_stat, p_val)
    return results


def log_statistics(
    stats: Dict[str, Tuple[float, float]],
    file_path: str
) -> None:
    """
    Write t-test statistics to a text file.

    Args:
        stats: Mapping of method to (t_stat, p_value).
        file_path: Path to the output log file.
    """
    with open(file_path, 'w') as f:
        f.write("Method\tT-statistic\tP-value\n")
        for method, (t_stat, p_val) in stats.items():
            f.write(f"{method}\t{t_stat:.6f}\t{p_val:.6f}\n")


def plot_depth_comparison(
    depths: List[int],
    results: Dict[str, float],
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot expectation values vs. depth for multiple methods.

    Args:
        depths: List of depths.
        results: Mapping of method name to list of expectation values.
        title: Optional plot title.

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots()
    for method, vals in results.items():
        ax.plot(depths, vals, marker='o', label=method)
    ax.set_xlabel('Depth')
    ax.set_ylabel('Expectation Value')
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return fig


def plot_ttest_pvalues(
    stats: Dict[str, Tuple[float, float]],
    title: Optional[str] = None
) -> plt.Figure:
    """
    Plot p-values from paired t-tests as a bar chart.

    Args:
        stats: Mapping of method to (t_stat, p_value).
        title: Optional plot title.

    Returns:
        Matplotlib Figure.
    """
    methods = list(stats.keys())
    pvalues = [stats[m][1] for m in methods]
    fig, ax = plt.subplots()
    ax.bar(methods, pvalues)
    ax.set_ylabel('P-value')
    if title:
        ax.set_title(title)
    return fig
