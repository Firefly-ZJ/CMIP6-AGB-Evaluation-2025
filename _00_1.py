#####     Pre-Processing     #####
import xarray as xr
import numpy as np
from joblib import load, dump
from skimage.transform import resize

import cartopy.crs as ccrs
import matplotlib.pyplot as plt    

def flipImage(image:np.ndarray):
    """Change image coordinate"""
    image = np.flip(image, axis=0)
    temp = np.zeros_like(image)
    temp[:, :180] = image[:, 180:]
    temp[:, 180:] = image[:, :180]
    image = temp
    return image

def resampleImage(image, shape:tuple=(180, 360)):
    """Resample image to given shape"""
    resized = resize(image, shape, order=1)
    print("Resized:", resized.shape)
    return resized

def resampleStd(image:xr.DataArray, ratio:int):
    """Resample std image, by given downsample ratio"""
    # std' = sqrt(Σ std^2) / n
    variance = image ** 2
    variance_sum = variance.coarsen(lat=ratio, lon=ratio, boundary="exact").sum()
    count = image.coarsen(lat=ratio, lon=ratio, boundary="exact").count()
    print(count.values)
    std_low_res = np.sqrt(variance_sum) / count
    print("Std Resized:", std_low_res.shape)
    return std_low_res

### ----- Resample ESA CCI dataset to 1d ----- ###
fileName = "./Data/ESACCI-BIOMASS-L4-AGB-MERGED-50000m-fv5.0.nc"
with xr.open_dataset(fileName) as dataSet:
    #print(dataSet)
    pass

with xr.open_dataset(fileName) as dataSet:
    val = dataSet["agb_sd"]
    print(val.shape)
    print(val)
    print()

    data = val[0,:,:]
    data = data / 2 ### AGB → Carbon
    print(f"Shape: {data.shape},  Range: {data.min():.2f} ~ {data.max():.2f}")
    
    data = resampleImage(data)  # resample to 1d
    #data = resampleStd(data, ratio=2)
    data = flipImage(np.asarray(data))  # change coordinate

    #lats, lons = len(val["lat"]), len(val["lon"])
    #lat, lon = np.arange(-90, 90.1, 180/lats), np.arange(0, 360.1, 360/lons)
    lat, lon = np.arange(-90, 90.1, 1), np.arange(0, 360.1, 1)
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines()
    img = ax.pcolormesh(lon, lat, data, transform=ccrs.PlateCarree(), shading="flat")
    plt.colorbar(img, orientation="horizontal", shrink=0.8, pad=0.05)
    plt.show(img)

    #dump(data, "./ESACCI-std_2010.joblib")