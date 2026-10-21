#####     Plot (every model)     #####
import numpy as np
from joblib import load
from osgeo import gdal
gdal.UseExceptions()
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm

import _calGlobal
getGlobal = lambda x: _calGlobal.getGlobalSum(x, factor=1e-7) # unit: PgC

lat, lon = np.arange(-90, 90.1, 1), np.arange(0, 360.1, 1)

def get_lat_mean(arr:np.ndarray, group=3) -> np.ndarray:
    """按纬度取平均.
    Input=[lat, lon]; Output=[lat/group]
    """
    lat, lon = arr.shape
    if lat % group != 0: raise ValueError("lat %% group != 0")
    reshaped = arr.reshape(lat//group, -1)
    return np.nanmean(reshaped, axis=-1)

### ---------- Load Data ---------- ###
ESA_data:np.ndarray = load("./Data/ESACCI_2010.joblib") # (lat, lon)
CMIP_set:dict = load("./Data/CMIPdata_5yr.joblib") # cVeg & treeFrac
CMIP_list = [i for (i,_) in CMIP_set.values()]
modelNum = len(CMIP_set) # 模型数量：30
modelNames = list(CMIP_set.keys())
print("Models: ", modelNum)

ValidMask = load("./Data/mask/validmask.joblib")
TreeMask = load("./Data/mask/treemask.joblib")
MASK = ValidMask & TreeMask ###
ESA_data[~MASK] = np.nan
for i in CMIP_list: i[~MASK] = np.nan
CMIP_data = np.stack(CMIP_list, axis=0) # CMIP data (N, lat, lon)
CMIP_mean = np.nanmean(CMIP_data, axis=0)
diff_mean = CMIP_mean - ESA_data

### ---------- Plot ---------- ###
def plot(data:np.ndarray, title="", diff_max=60., save_fig=False, apply_corr=True):
    fig = plt.figure(figsize=(16, 8.5), dpi=120)
    gs = GridSpec(2, 3, fig, width_ratios=[1, 1, 0.4], hspace=0.05)

    # ========== 上半部分子图 ========== #
    ### 左上：观测值 Map
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson())
    img1 = ax1.pcolormesh(lon, lat, ESA_data, vmin=0, cmap="YlGn", transform=ccrs.PlateCarree(), shading="flat")
    ax1.set_global()
    ax1.coastlines(color="gray")
    ax1.set_title('(a) Carbon Density (ESA CCI)')
    fig.colorbar(img1, ax=ax1, orientation="horizontal", shrink=0.8, pad=0.05)

    ### 右上：CMIP mean Map
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.Robinson())
    img2 = ax2.pcolormesh(lon, lat, data, vmin=0, cmap="YlGn", transform=ccrs.PlateCarree(), shading="flat")
    ax2.set_global()
    ax2.coastlines(color="gray")
    ax2.set_title('(b) Carbon Density (CMIP6 model)')
    fig.colorbar(img2, ax=ax2, orientation="horizontal", shrink=0.8, pad=0.05)

    ### 纬度统计图
    group = 5 ### 5°一个点
    lat_range = np.arange(-90+group/2, 90, group)
    ESA_lat = get_lat_mean(ESA_data, group=group)
    CMIP_lat = get_lat_mean(data, group=group)

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(ESA_lat, lat_range, linestyle="-", color="gray", label="ESA CCI")
    ax3.plot(CMIP_lat, lat_range, linestyle="-", color="orange", label="CMIP6")
    pos1, pos3 = ax1.get_position(), ax3.get_position()
    ax3.set_position([pos3.x0, pos1.y0, pos3.width, pos1.height])
    ax3.set_ylim(-90, 90)
    ax3.set_yticks(np.arange(-60, 61, 30))
    ax3.legend(loc="best", fontsize=8, handlelength=1)
    ax3.set_title('(c) Latitude Mean')

    # ========== 下半部分子图 ========== #
    ### 左下：绝对值 Difference
    data1 = data - ESA_data
    ax4 = fig.add_subplot(gs[1, 0], projection=ccrs.Robinson())
    img4 = ax4.pcolormesh(lon, lat, data1, vmin=-diff_max, vmax=diff_max, cmap="RdBu_r", transform=ccrs.PlateCarree(), shading="flat")
    ax4.set_global()
    ax4.coastlines(color="gray")
    ax4.set_title('(d) Difference (abs)')
    fig.colorbar(img4, ax=ax4, orientation="horizontal", shrink=0.8, pad=0.05)

    ### 右下：Diff 一致性
    cons = np.full_like(data1, np.nan)
    cons[(data1>0) & (diff_mean>0)] = 1.5
    cons[(data1>0) & (diff_mean<0)] = 0.5
    cons[(data1<0) & (diff_mean>0)] = -0.5
    cons[(data1<0) & (diff_mean<0)] = -1.5
    # colormap
    colors = [(0.,0.,0.7), (0.6,0.6,1.), (1.,0.6,0.6), (0.7,0.,0.)]
    labels = ['Underestimate', '', '', 'Overestimate']
    cmap = ListedColormap(colors)
    bounds = list(range(-2, 3))
    norm = BoundaryNorm(bounds, cmap.N)

    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.Robinson())
    img5 = ax5.pcolormesh(lon, lat, cons, cmap=cmap, norm=norm, transform=ccrs.PlateCarree(), shading="flat")
    ax5.set_global()
    ax5.coastlines(color="gray")
    ax5.set_title('(e) Consistence')
    cbar = fig.colorbar(img5, ax=ax5, orientation="horizontal", shrink=0.6, pad=0.05)
    cbar.set_ticks([-1.5, -0.5, 0.5, 1.5])
    cbar.set_ticklabels(labels)

    ### 纬度统计图
    ax6 = fig.add_subplot(gs[0, 2])
    ax6.axvline(x=0, color="gray", linestyle="--")
    ax6.plot(CMIP_lat-ESA_lat, lat_range, linestyle="-", color="orangered")
    pos4, pos6 = ax4.get_position(), ax6.get_position()
    ax6.set_position([pos6.x0, pos4.y0, pos6.width, pos4.height])
    ax6.set_ylim(-90, 90)
    ax6.set_yticks(np.arange(-60, 61, 30))
    ax6.set_title('(f) Diff Mean')

    if title != "": plt.suptitle(title, fontsize=14, y=0.9, va="center")
    if not save_fig: plt.show(fig)
    else: plt.savefig(f"./250623/models/{title}.png", bbox_inches="tight")
    plt.close(fig)

    # ========== 相关性及校正 ========== #
    if not apply_corr: return
    validMask = (ESA_data > 0) & (data > 0)
    X = ESA_data[validMask].reshape(-1, 1)
    Y = data[validMask]
    model = LinearRegression().fit(X, Y)
    slope, intercept = model.coef_, model.intercept_
    r2 = model.score(X, Y)
    func = lambda x: slope * x + intercept

    fig, ax = plt.subplots()
    ax.scatter(X.flatten(), Y, c="gray", alpha=0.2)
    xrange = np.linspace(0, np.max(X), 100)
    ax.plot(xrange, func(xrange), c="red", linewidth=3)
    ax.plot(xrange, xrange, linestyle="--", color="black")
    ax.set_xlabel("ESA CCI")
    ax.set_ylabel(title)

    text = f"$y = {slope[0]:.2f} x + {intercept:.2f}$\nR$^2$ = {r2:.2f}"
    ax.text(0.05, 0.85, text, fontsize=12, transform=ax.transAxes)
    
    plt.suptitle(title, fontsize=14, y=0.9, va="center")
    plt.show(fig)
    plt.close(fig)

### ---------- Main ---------- ###
ESA_2015 = load("./Data/ESACCI_2015.joblib")
ESA_2015[~MASK] = np.nan
plot(ESA_2015, title="ESA CCI 2015", diff_max=10)

for i in range(len(modelNames)):
    print(f"\n{i}:  {modelNames[i]}")
    plot(CMIP_data[i], title=modelNames[i], save_fig=False, apply_corr=False)
print("Completed")