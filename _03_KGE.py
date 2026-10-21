#####     KGE and Clusteing     #####
import numpy as np
from joblib import load
from osgeo import gdal
gdal.UseExceptions()
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from scipy.stats import pearsonr
def cal_KGE(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """Calculate the Kling-Gupta Efficiency of prediction.
    KGE = 1 - sqrt{(r-1)^2 + (alpha-1)^2 + (beta-1)^2}
    """
    r, p = pearsonr(y_true, y_pred)  # 相关系数
    a = np.std(y_pred) / np.std(y_true)  # 变异性比率
    b = np.mean(y_pred) / np.mean(y_true)  # 偏差比率
    temp = (r-1)**2 + (a-1)**2 + (b-1)**2
    return 1 - np.sqrt(temp)

### ---------- Load Data ---------- ###
ESA_data:np.ndarray = load("./Data/ESACCI_mean.joblib") # (lat, lon)
CMIP_set:dict = load("./Data/CMIPdata_5yr.joblib") # cVeg & treeFrac
CMIP_list = [i for (i,_) in CMIP_set.values()]
CMIP_data = np.stack(CMIP_list, axis=0) # CMIP data (N, lat, lon)
modelNum = len(CMIP_set) # 模型数量：30
modelNames = list(CMIP_set.keys())
print("Models: ", modelNum)

ValidMask = load("./Data/Masks/validmask.joblib")
TreeMask = load("./Data/Masks/treemask.joblib")
MASK = ValidMask & TreeMask ###

### climate zone mask
Koppen:np.ndarray = gdal.Open("./Data/Koppen_2010.tif", gdal.GA_ReadOnly).ReadAsArray().astype(np.float32)
Koppen = np.flip(Koppen, axis=0)
temp = np.zeros_like(Koppen)
temp[:,:180] = Koppen[:,180:]
temp[:,180:] = Koppen[:,:180]
Koppen = temp

cliDict = {
    "Af": [1],  # 热带雨林
    "Am": [2],  # 热带季风
    "Aw": [3],  # 热带萨瓦纳
    "BW": [4, 5], # 干旱沙漠
    "BS": [6, 7], # 半干旱草原
    "Cs": [8, 9, 10], # 亚热带夏干（地中海）
    "Cw": [11, 12, 13], # 亚热带冬干
    "Cf": [14, 15, 16], # 亚热带全湿
    "Ds": [17, 18, 19, 20], # 寒温带夏干
    "Dw": [21, 22, 23, 24], # 寒温带冬干
    "Df": [25, 26, 27, 28], # 寒温带全湿
    "ET": [29],  # 极地苔原
    "EF": [30]   # 极地冰原
    }
# f=全湿，s=夏干，w=冬干
cliMask = [np.isin(Koppen, ids) for ids in cliDict.values()] ### Climate Mask
# "Af", "Am", "Aw", "BW", "BS",
# "Cs", "Cw", "Cf", "Ds", "Dw", "Df", "ET", "EF"
ZoneMasks = [np.full_like(MASK, True), # Global mask
             cliMask[0] | cliMask[1] | cliMask[2],  # Tropical
             cliMask[5] | cliMask[6] | cliMask[7],  # Sub-tropical
             cliMask[8] | cliMask[9] | cliMask[10]  # Frigid-temperate
             ]

### ---------- Calculate KGE ---------- ###
### Calculate KGE for each model
print("\nCalculate KGE for each model")
CMIP_mean = np.nanmean(CMIP_data, axis=0)
print("MME KGE:", end=" ")
for m in ZoneMasks:
    mask1 = MASK & m
    KGE = cal_KGE(ESA_data[mask1], CMIP_mean[mask1])
    print(f"{KGE:.4f}", end=",  ")
# MME KGE: 0.7202,  0.6328,  0.0600,  0.4067
KGE_list = []
for i in range(modelNum):
    modelName = modelNames[i]
    print(f"\n{i:02d} {modelName:<16s}", end=" ")
    modelData = CMIP_data[i]  # (lat, lon)
    result = []
    mask0 = MASK & (modelData>0)
    for m in ZoneMasks:
        mask1 = mask0 & m
        KGE = cal_KGE(ESA_data[mask1], modelData[mask1])
        result.append(KGE)
        print(f"{KGE:.4f}", end=",  ")
    KGE_list.append(result)
print()

### Cluster models by KGE
print("\nCluster models by KGE")
#from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
#from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

data = np.array(KGE_list)  # (modelNum, zoneNum)
#scaler = StandardScaler() # 数据标准化
#data = scaler.fit_transform(data)

### 自动选择最佳聚类数（基于轮廓系数）
def find_optimal_clusters(data, min_k=2, max_k=10):
    if not (1 <= min_k <= max_k): raise ValueError("Invalid K range")
    silhouette_scores = []
    for k in range(min_k, max_k+1):
        model = AgglomerativeClustering(n_clusters=k)
        labels = model.fit_predict(data)
        score = silhouette_score(data, labels)
        silhouette_scores.append(score)
    optimal_k = np.argmax(silhouette_scores) + min_k
    max_score = np.max(silhouette_scores)
    return optimal_k, max_score

optimal_k, max_score = find_optimal_clusters(data, min_k=2, max_k=10)
print(f"Optimal K num: {optimal_k},  Max score: {max_score}")

optimal_k = 4  ### chose 4 as k
final_model = AgglomerativeClustering(n_clusters=optimal_k)
cluster_result = final_model.fit_predict(data)

### ---------- PCA可视化结果 ---------- ###
use_PCA_ax = True ###
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data)
if use_PCA_ax:
    comX, comY = principal_components[:, 0], principal_components[:, 1]
else: comX, comY = data[:, 1], data[:, 2]

fig, ax = plt.subplots(figsize=(9, 4), dpi=150)
"""
img = ax.scatter(comX, comY, s=40, c=cluster_result, marker="o")
plt.colorbar(img, orientation="vertical")
#for i in range(modelNum): ax.annotate(str(i), (comX[i], comY[i]), xytext=(3,4), textcoords='offset points', fontsize=10)
"""
colors = [(0.1, 0.1, 1.), (0.1, 1., 0.1), (1., 0.1, 0.1)]
cmap = LinearSegmentedColormap.from_list("mycmap", colors)
markers = ['o', 's', '^', '*', 'D'] # 圆圈、方块、三角形、星星、菱形
for i in range(modelNum):
    ax.scatter(comX[i], comY[i], s=40, color=cmap(i/modelNum), alpha=0.7,
               marker=markers[i%len(markers)], label=modelNames[i])
ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), ncol=2, fontsize=9)

if use_PCA_ax:
    ax.set_xlim(-2, 1.5)
    ax.set_ylim(-1.5, 1.5)
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    feature_names = ["Global", "Tropical", "Temperate", "Boreal"]
    loadings = pca.components_.T  # 转置后每行对应一个原始特征
    arrow_scale = 1.2  # 调整箭头长度比例因子
    shifts = [(0.25, -0.1), (0.15, -0.15), (0.15, 0.15), (-0.2, 0.1)]
    for loading, name, (shiftx, shifty) in zip(loadings, feature_names, shifts):
        dx, dy = loading[0] * arrow_scale, loading[1] * arrow_scale
        ax.arrow(0, 0, dx, dy, color="dimgray", alpha=0.9, linewidth=1.5, head_width=0.08, head_length=0.08)
        ax.text(dx+shiftx, dy+shifty, name, fontsize=10, ha='center', va='center')
else:
    plt.xlabel("KGE Tropical")
    plt.ylabel("KGE Temperate")

#ax.set_title(f"Agglomerative Clustering (K={optimal_k})")
ax.axhline(0, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
ax.axvline(0, color='gray', alpha=0.3, linewidth=0.5, linestyle='--')
plt.tight_layout()
plt.show()
plt.close()

print("\nClustering Result:")
for i in range(optimal_k):
    group = [modelNames[j] for j in range(modelNum) if cluster_result[j] == i]
    print(f"Cluster {i} (N={len(group)}):  {',  '.join(group)}")

print("Completed")

# Cluster 0 (N=7):  ACCESS-ESM1-5,  CMCC-CM2-SR5,  CMCC-ESM2,  CNRM-ESM2-1,  TaiESM1,  UKESM1-0-LL,  UKESM1-1-LL
# Cluster 1 (N=19):  BCC-CSM2-MR,  BCC-ESM1,  CanESM5,  CESM2-FV2,  CESM2-WACCM-FV2,  CESM2-WACCM,  CESM2,  E3SM-1-1-ECA,  EC-Earth3-CC,  EC-Earth3-Veg-LR,  GFDL-ESM4,  INM-CM4-8,  INM-CM5-0,  IPSL-CM6A-LR-INCA,  IPSL-CM6A-LR,  MPI-ESM-1-2-HAM,  MPI-ESM1-2-LR,  NorESM2-LM,  NorESM2-MM
# Cluster 2 (N=2):  E3SM-1-1,  SAM0-UNICON
# Cluster 3 (N=2):  KIOST-ESM,  NorCPM1
