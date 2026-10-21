#####    Taylor Diagram     #####
"""Taylor diagram (Taylor, 2001) implementation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections import PolarAxes

class TaylorDiagram():
    """
    Taylor diagram.

    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with radius=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd:float, fig:plt.Figure,
                 rect=111, std_range=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.

        Args:
            refstd: reference standard deviation
            fig: input Figure
            rect: subplot position
            std_range: stddev axis extension (times of *refstd*)
            extend: extend diagram to negative correlations
        """

        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation
        self.samplePoints = []          # List of sample points

        ptr = PolarAxes.PolarTransform(apply_theta_transforms=False)

        # Correlation labels
        rlocs = np.array([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = std_range[0] * self.refstd
        self.smax = std_range[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            ptr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set(text="Correlation", fontsize=12, axis_direction="top")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set(text="Standard deviation", fontsize=12)

        ax.axis["right"].set_axis_direction("top")    # "Y axis"
        ax.axis["right"].toggle(ticklabels=True, label=True)
        ax.axis["right"].label.set(text="Standard deviation", fontsize=12)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)       # Unused

        self._ax = ax                    # Graphical axes
        self.ax = ax.get_aux_axes(ptr)   # Polar coordinates
        self._ax.axis[:].major_ticks.set_tick_out(True) # Put ticks outward

        # Add reference point and stddev contour
        refPoint = self.ax.scatter([0], [self.refstd],
                    label="Ref", marker='o', s=20, c='k')
        t = np.linspace(0, self.tmax) # theta
        r = np.zeros_like(t) + self.refstd # radius
        self.ax.plot(t, r, 'k--')
        self.samplePoints.append(refPoint)

    def add_sample(self, stddev:float|np.ndarray, corrcoef:float|np.ndarray, **scatter_kwargs):
        """Add a sample (*stddev*, *corrcoef*) point to the Taylor diagram."""
        stddev, corrcoef = np.asarray(stddev), np.asarray(corrcoef)
        theta = np.arccos(corrcoef)
        samplePoint = self.ax.scatter(theta, stddev, **scatter_kwargs)  # (theta, radius)
        self.samplePoints.append(samplePoint)
        return samplePoint
    
    def get_handles(self):
        """Return handles for legend."""
        return self.samplePoints

    def add_grid(self, color="0.75", alpha=0.8, **grid_kwargs):
        """Add a grid."""
        self._ax.grid(True, color=color, alpha=alpha, **grid_kwargs)

    def add_RMSD(self, levels=5, colors="0.75", linestyles="--", alpha=0.8,
                 inline=True, fontsize=10, **kwargs):
        """
        Add constant centered RMSD contours, defined by *levels*.

        - Line Args: levels, colors, linestyles, alpha;
        - Label Args: inline, fontsize.
        """
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels,
                colors=colors, linestyles=linestyles, alpha=alpha, **kwargs)
        self._ax.clabel(contours, inline=inline, fontsize=fontsize, **kwargs)

### ----- test ----- ###
def test():
    # Reference std & Samples
    stdref = 48.491
    samples = [[25.939, 0.365, "Model A"],
               [29.593, 0.509, "Model B"],
               [33.125, 0.585, "Model C"],
               [29.593, 0.659, "Model D"],
               [61.215, 0.473, "Model E"]
               ]
    
    # Create Taylor diagram
    fig = plt.figure(figsize=(8,6))
    dia = TaylorDiagram(stdref, fig, extend=False)
    #dia.samplePoints[0].set_color('r')  # Mark reference point as a red star

    # Add models to Taylor diagram
    for i, (stddev, corrcoef, name) in enumerate(samples):
        dia.add_sample(stddev, corrcoef, label=name,
                       marker='$%d$' % (i+1), s=60, c='k')

    # Add RMSD contours and grid
    dia.add_grid()
    dia.add_RMSD(levels=5)

    # Add a figure legend and title
    fig.legend(dia.samplePoints, [p.get_label() for p in dia.samplePoints],
               loc='upper right', bbox_to_anchor=(0.9, 0.9))
    fig.suptitle("Taylor diagram", fontsize="x-large")

    plt.show(fig)
    plt.close(fig)


if __name__ == '__main__':
    test()