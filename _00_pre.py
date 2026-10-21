#####     Pre-Processing     #####
import os
import re
import xarray as xr
import numpy as np
from joblib import dump
from skimage.transform import resize

import cartopy.crs as ccrs
import matplotlib.pyplot as plt

import _calGlobal
getGlobal = _calGlobal.getGlobalSum

rootPath = "./historical/"

resizeShape = (180, 360)  # (lat, lon): 1°*1°
lat, lon = np.arange(-90, 90.1, 1), np.arange(0, 360.1, 1)

def findMatching(pattern:str, files:list):
    for file in files:
        if re.match(pattern, file): return file
    return None

### ---------- CMIP6 数据预处理 ---------- ###
print("Data preprocessing...")
cVegData, cRootData = [], []
treeData = []
rootNum, treeNum = 0, 0

for f in os.walk(rootPath):  # 遍历文件夹
    if not f[0] == rootPath: continue
    else: files = f[2]
for fileName in files:
    keywords = fileName.split("_")
    if not keywords[0] == "cVeg": continue
    else:
        # cVeg数据
        cVegData.append(fileName)
        # cRoot数据
        pattern = "cRoot_" + "_".join(keywords[1:4]) + ".*nc"
        cRootName = findMatching(pattern, files)  ### 使用正则表达式
        cRootData.append(cRootName)
        if cRootName: rootNum += 1
        # treeFrac数据
        pattern = "treeFrac_" + "_".join(keywords[1:4]) + ".*nc"
        treeName = findMatching(pattern, files)
        treeData.append(treeName)
        if treeName: treeNum += 1

print("cVeg Data: ", len(cVegData))
print(cVegData)
print("cRoot Data: ", rootNum)
print(cRootData)
print("treeFrac Data: ", treeNum)
print(treeData)
print()

allData = dict()

### Read data
for idx, fileName in enumerate(cVegData):
    period = fileName.split("_")[-1].split(".")[0]
    model = fileName.split("_")[2]
    print(f"{idx}:  {model},  {period}")
    with xr.open_dataset(rootPath+fileName) as dataSet:
        cVeg = dataSet["cVeg"]
        print("Shape: ", cVeg.shape)
        print("Dims: ", cVeg.dims)
        #print(cVeg)
        cVegVal = np.mean(np.array(cVeg[-60:,:,:]), axis=0)  # 5年平均
    
    ### cAGB = cVeg - cRoot
    if not cRootData[idx]:
        print("No cRoot data")
        AGBcarbon = cVegVal * 0.8 ### 4:1
    else:
        with xr.open_dataset(rootPath+cRootData[idx]) as dataSet:
            cRoot = dataSet["cRoot"]
            cRootVal = np.mean(np.array(cRoot[-60:,:,:]), axis=0)  # 5年平均
        AGBcarbon = cVegVal - cRootVal
    AGBcarbon[AGBcarbon < 0] = 0
    AGBcarbon = AGBcarbon * 10  # Unit: kgC/m2 → MgC/ha
    if idx in [11, 12]:  ### 原始数据单位问题!
        AGBcarbon = AGBcarbon / 1000
    
    ### treeFrac
    treeFracVal = None
    if not treeData[idx]:
        print("No treeFrac data")
        print("Rawdata SUM:", getGlobal(AGBcarbon, mask=None, factor=1e-7))
    else:
        with xr.open_dataset(rootPath+treeData[idx]) as dataSet:
            treeFrac = dataSet["treeFrac"]
            treeFracVal = np.mean(np.array(treeFrac[-60:,:,:]), axis=0)  # 5年平均
            treeFracVal = treeFracVal / 100  ###
            print("Rawdata SUM2:", getGlobal(AGBcarbon, mask=treeFracVal, factor=1e-7))

    ### Resize
    AGBcarbon[AGBcarbon <= 0] = np.nan
    AGBcarbon = resize(AGBcarbon, resizeShape, order=0)  ### 最近邻插值
    if treeFracVal is not None:
        treeFracVal[treeFracVal <= 0] = np.nan
        treeFracVal = resize(treeFracVal, resizeShape, order=0)  ### 最近邻插值
    
    ### Save Data
    if model in allData: raise RuntimeWarning("Same Name Data: ", model)
    allData[model] = (AGBcarbon, treeFracVal) # 保存

    print(f"MAX: {np.nanmax(AGBcarbon):.2f} MgC/ha,  SUM: {getGlobal(AGBcarbon, factor=1e-7):.2f} PgC")
    if treeFracVal is not None:
        print(f"Tree: {np.nanmean(treeFracVal) * 100:.2f}%,  "+
              f"SUM2: {getGlobal(AGBcarbon, mask=treeFracVal, factor=1e-7):.2f} PgC")

    ### plot
    val = AGBcarbon*treeFracVal if treeFracVal is not None else AGBcarbon
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.coastlines(color="gray")
    img = ax.pcolormesh(lon, lat, val, cmap="YlGn",
                        vmin=0, transform=ccrs.PlateCarree(), shading="flat")
    plt.title(model if treeFracVal is None else model+" *", fontsize=12)
    plt.colorbar(img, orientation="horizontal", pad=0.05, shrink=0.8)
    plt.show(img)
    plt.close()
    print()

dump(allData, "./CMIPdata_5yr.joblib")  ### allData[model] = (AGBcarbon, treeFracVal)
print("Processing completed")
print(len(allData), list(allData.keys()))