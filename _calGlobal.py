#####     Calculate Global Sum     #####
import numpy as np

Radius = 6371.0  ### 地球半径 Earth Mean Radius (km)
Pi = np.pi
Sin = np.sin

def getArea(lat, latRes, lonRes=0.):
    """Calculate pixel area at given latitude (km^2)
    Args:
        lat (float): 纬度位置 (south-side, degree)
        latRes (float): 纬度分辨率 (degree)
        lonRes (float, optional): 经度分辨率 (degree, default equal to latRes)
    """
    if not lonRes: lonRes = latRes
    alpha = lat * Pi/180
    beta = (lat + latRes) * Pi/180
    return Radius**2 * (lonRes * Pi/180) * (Sin(beta) - Sin(alpha))

def getGlobalSum(data:np.ndarray, mask=None, factor:float=1., is_std:bool=False) -> float:
    """Calculate global sum of given data (np.nan → 0) (copy data first)
    Args:
        data (np.ndarray): 输入数据 (in shape of [lat, lon])
        mask (np.ndarray, optional): 输入数据掩膜
        factor (float): 单位换算系数: output = factor*sum
        is_std (bool): 输入是std而不是data
            - 假设误差相互独立, 方差相加再开方: STD = Sqrt(Σ std^2)
            - 若误差不相互独立, 默认`is_std=False`, 直接计算总和
    """
    globalSum = 0.
    data = data.copy() ### 避免修改输入数据
    shape = data.shape
    if mask is not None:
        if not (mask.shape == shape): raise ValueError("Mismatching mask shape")
        else: data = data * mask
    latRes, lonRes = 180/shape[0], 360/shape[1]
    lats = np.arange(-90, 90, step=latRes)

    for (line, lat) in zip(data, lats):
        temp = line[~np.isnan(line)]
        if len(temp) == 0: continue
        pixArea = getArea(lat, latRes, lonRes)
        if not is_std:
            globalSum += np.sum(temp) * pixArea
        else:
            globalSum += np.sum((temp*pixArea)**2)  # 方差
    
    if is_std: globalSum = np.sqrt(globalSum)
    return globalSum * factor

if __name__ == "__main__":
    print("Earth Surface: 510M km^2")
    print(getArea(-90, 180, 360) / 1e6)
    print(getArea(0, 1) / 1e6)

    data = np.ones((180, 360), dtype=np.float32)
    print(getGlobalSum(data, mask=(data>0), factor=1e-6))