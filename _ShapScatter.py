#####     Custom SHAP Scatter Plot     #####
from typing import Any, Literal, Iterable
import matplotlib.pyplot as plt
import numpy as np

import shap
from shap.plots import scatter
from shap._explanation import Explanation
from shap.plots import colors
from shap.plots._utils import AxisLimitSpec, parse_axis_limit


# TODO: Make the color bar a one-sided beeswarm plot so we can see the density along the color axis
def shap_scatter(
    shap_values: Explanation,
    features: Iterable,
    fig: plt.Figure,
    subplots: np.ndarray[plt.Axes],
    color: str | Explanation | None = "#1E88E5",
    hist: bool = True,
    cmap = colors.red_blue,
    twin_yaxis: bool = True,
    y_thr: float|None = None,
    alpha: float = 1.0,
    title: str | None = None,
    overlay: dict[str, Any] | None = None,
    ylabel: str = "SHAP value",
):
    """Create a SHAP dependence scatter plot, optionally colored by an interaction feature.
    
    It will use the :func:`shap.plots.scatter` function to draw each subplot.
    It's recommended to use :func:`shap.plots.scatter` directly when draw a single feature.

    Parameters
    ----------
    shap_values (shap.Explanation)
        Typically a single column of an :class:`.Explanation` object
        (i.e. `shap_values[:, "Feature A"]`).

        Alternatively, pass multiple columns to create several subplots
        (i.e. `shap_values[:, ["Feature A", "Feature B"]]`).
    
    features (Iterable, Custom)
        Names of the features need to plot.
    
    fig (plt.Figure, Custom)
        The figure to draw the plot on.

    subplots (np.ndarray[plt.Axes], Custom)
        The subplots of figure to draw the plot on. Its shape should match the
        number of columns in `shap_values`.
    
    color (string or shap.Explanation, optional)
        How to color the scatter plot points. This can be a fixed color string, or an
        :class:`.Explanation` object.

        If it is an :class:`.Explanation` object, then the scatter plot points are
        colored by the feature that seems to have the strongest interaction effect with
        the feature given by the ``shap_values`` argument. This is calculated using
        :func:`shap.utils.approximate_interactions`.

        If only a single column of an :class:`.Explanation` object is passed, then that
        feature column will be used to color the data points.

    hist (bool, optional)
        Whether to show a light histogram along the x-axis to show the density of the data.
        Default is `True`.
    
    twin_yaxis (bool, optional)
        Whether to use a twin y-axis for each raw of subplots. Default is `True`.
    
    y_thr (float|None, optional)
        The minimum range of the y-axis (larger than 0). If None, then the y-axis will be auto-scaled.

    overlay (dict|None, optional)
        Dictionary of up to three additional curves to overlay as line plots.
        
    Returns
    -------
    fig (plt.Figure)
        Returns the :class:`plt.Figure` object with the plot drawn onto it.
    """
    if not isinstance(shap_values, Explanation):
        raise TypeError("The shap_values parameter must be a shap.Explanation object!")
    if len(features) == 1:  ###
        print("Note: Recommend to use shap.plots.scatter directly, when only 1 feature.")
    if len(features) > subplots.size:   
        raise ValueError(f"Too many features to plot! Only {subplots.size} subplots available.")
    
    # Plot each subplot
    if len(subplots.shape) == 1: n_cols, n_rows = subplots.shape[0], 1
    else: n_rows, n_cols = subplots.shape
    for i, name in enumerate(features):  ### i = subplot index, name = feature name
        # auto y-axis limits
        if twin_yaxis:
            same_row = [features[j + (i//n_cols)*n_cols] for j in range(n_cols)]
            ax_values = np.array(shap_values[:, same_row].values)
            ymin, ymax = np.min(ax_values), np.max(ax_values)
        else:
            ymin, ymax = np.min(shap_values[:, name].values), np.max(shap_values[:, name].values)
        if (y_thr is not None) and (y_thr > 0):
            ymin, ymax = min(ymin, -y_thr), max(ymax, y_thr)
        buffer = (ymax - ymin) / 10
        ymin, ymax = ymin - buffer, ymax + buffer
        # plot a single scatter plot
        ax:plt.Axes = subplots[i] if n_rows == 1 else subplots[i//n_cols, i%n_cols]
        scatter(shap_values[:, name], ax=ax, show=False,  ###
                color=color, hist=hist, cmap=cmap, alpha=alpha, ymin=ymin, ymax=ymax)
        # plot overlay lines
        if overlay is not None:
            line_styles = ["solid", "dotted", "dashed"]
            for j, (name, vals) in enumerate(overlay.items()):
                if isinstance(vals[i][0][0], (float, int)):
                    ax.plot(vals[i][0], vals[i][1], color="#000000", linestyle=line_styles[j], label=name)
        # set axis labels and index text
        ax.text(0.9, 0.9, f"({chr(97+i)})", transform=ax.transAxes, fontsize=14)
        if i % n_cols == 0:
            ax.set_ylabel(ylabel)
        else:
            ax.set_ylabel("")
            if twin_yaxis:
                ax.set_yticks([])
                ax.spines["left"].set_visible(False)
            else: pass
        
    # Add title and legend
    if title is not None:
        fig.suptitle(title)
    if overlay is not None:
        plt.legend()
    return fig

"""
# Example:
fig, axs = plt.subplots(3, 4, figsize=(20, 12), dpi=150)
fig = shap_scatter(expl, names[0:12], fig, axs, color=expl,
        hist=True, twin_yaxis=False, y_thr=10, alpha=0.8)
plt.show(fig)
"""

if __name__ == "__main__":
    pass
