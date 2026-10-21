#####     Pre-Processing     #####
import numpy as np
from joblib import load
from osgeo import gdal
gdal.UseExceptions()

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

from _calGlobal import getGlobalSum
getGlobal = lambda x: getGlobalSum(x, factor=1e-7) # unit: PgC

CMIP_data:dict = load("./Data/CMIPdata_5yr.joblib")
ESA_data:np.ndarray = load("./Data/ESACCI_2010.joblib")
Koppen:np.ndarray = gdal.Open("./Data/Koppen_2010.tif", gdal.GA_ReadOnly).ReadAsArray()

### 处理koppen地图坐标系
Koppen = np.flip(Koppen, axis=0)
temp = np.zeros_like(Koppen)
temp[:,:180] = Koppen[:,180:]
temp[:,180:] = Koppen[:,:180]
Koppen = temp
lat, lon = np.arange(-90, 90.1, 1), np.arange(0, 360.1, 1)
ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()
ax.pcolormesh(lon, lat, Koppen, vmin=0, vmax=30,
              transform=ccrs.PlateCarree(), shading="flat")
plt.title("Koppen Climate Zone")
plt.show()
plt.close()

### Koppen Climate Mask
MASK = Koppen > 0
cli_dict = {
    "Af": [1],
    "Am": [2],
    "Aw": [3],
    "Bw": [4, 5],
    "Bs": [6, 7],
    "Cs": [8, 9, 10],
    "Cw": [11, 12, 13],
    "Cf": [14, 15, 16],
    "Ds": [17, 18, 19, 20],
    "Dw": [21, 22, 23, 24],
    "Df": [25, 26, 27, 28],
    "Et": [29],
    "Ef": [30]
}
cliMask = []
for idxs in cli_dict.values():
    mask = np.isin(Koppen, np.array(idxs))
    cliMask.append(mask)

### Tree Mask
tree_thr = 0.5 # Tree Fraction threshold
print("Tree Fraction Threshold: ", tree_thr)
treeFrac = np.stack([tf for (_, tf) in CMIP_data.values() if tf is not None], axis=0)
treeFrac[np.isnan(treeFrac)] = 0.
treeFrac = np.mean(treeFrac, axis=0)
treeFrac[treeFrac==0] = np.nan
treeMask = treeFrac > tree_thr ###

ax = plt.axes(projection=ccrs.Robinson())
ax.set_global()
ax.coastlines()
ax.pcolormesh(lon, lat, treeFrac, cmap="Greens",
              vmin=0., vmax=1., transform=ccrs.PlateCarree(), shading="flat")
plt.title("Tree Fraction")
plt.show()
plt.close()

### ESACCI Data
data10, data1t = [], []
ESA_data[~MASK] = 0.
print(f"ESACCI Carbon:  {getGlobal(ESA_data):.2f} PgC")
data10.append(getGlobal(ESA_data))
data1t.append(getGlobal(ESA_data * treeMask))
for (cli, mask) in zip(cli_dict.keys(), cliMask):
    temp = ESA_data.copy()
    temp[~mask] = 0.
    print(f"{cli} = {getGlobal(temp):.2f} / {getGlobal(temp * treeMask):.2f}")
    data10.append(getGlobal(temp))
    data1t.append(getGlobal(temp * treeMask))

### CMIP Data
data20, data2t = [], [] # Raw cAGB data; cAGB data * tree mask
print("\nCMIP Carbon:")
for i, (model, (cAGB0, tf)) in enumerate(CMIP_data.items()):
    aaa, bbb = [], []
    cAGB0[~MASK] = 0.
    cAGBt = cAGB0 * treeMask
    aaa.append(getGlobal(cAGB0))
    bbb.append(getGlobal(cAGBt))
    text0, textt = [], []
    for (cli, mask) in zip(cli_dict.keys(), cliMask):
        # raw cAGB
        temp = cAGB0.copy()
        temp[~mask] = 0.
        text0.append(f"{getGlobal(temp):.2f}")
        aaa.append(getGlobal(temp))
        # cAGB * treeFrac
        temp = cAGBt.copy()
        temp[~mask] = 0.
        textt.append(f"{getGlobal(temp):.2f}")
        bbb.append(getGlobal(temp))
    data20.append(aaa)
    data2t.append(bbb)
    print(model)
    print(",  ".join(text0))
    print(",  ".join(textt))
    print()

### Plot
fig = plt.figure(figsize=(12, 4))

x1 = np.arange(0, len(data10)) - 0.3
ESAbar0 = plt.bar(x1, data10, width=0.25, label="ESACCI")
ESAbart = plt.bar(x1, data1t, width=0.25, hatch="///", label="ESACCI *")

x2 = np.arange(0, len(data10)) + 0.
y2 = np.mean(data20, axis=0)
std2 = np.std(data20, axis=0)
CMIPbar = plt.bar(x2, y2, width=0.25, yerr=std2, label="CMIP6")

x3 = np.arange(0, len(data10)) + 0.3
y3 = np.mean(data2t, axis=0)
std3 = np.std(data2t, axis=0)
CMIPbar = plt.bar(x3, y3, width=0.25, yerr=std3, label="CMIP6 *")

plt.xticks(np.arange(0, len(data10)), ["Global"]+list(cli_dict.keys()))
plt.legend()
plt.title("Total Carbon")
plt.show(fig)
plt.close(fig)