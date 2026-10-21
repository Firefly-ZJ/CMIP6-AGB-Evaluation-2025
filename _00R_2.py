#####     CMIP6 Preprocessing     #####
# 读取 _00R_1.py 生成的 2010~2014 年平均、1° 分辨率 NetCDF 文件，
# 再完成单位转换、无效值处理和补充变量计算，供后续分析使用。
from pathlib import Path
from collections import OrderedDict
import warnings

import numpy as np
import xarray as xr
from joblib import dump
import matplotlib.pyplot as plt


ROOT_PATH = Path("./Data/Resampled/")
RAW_OUT_PATH = Path("./CMIP6_raw.joblib")
SUP_OUT_PATH = Path("./CMIP6_sup.joblib")

RESAMPLED_SHAPE = (180, 360)
FLUX_SCALE_FACTOR = 10000.0  # kgC m-2 s-1 -> kgC ha-1 s-1

### !!! 以下模型的原始文件存在已知单位差异 !!!
SPECIAL_CARBON_UNIT = {"E3SM-1-1", "E3SM-1-1-ECA"}
SPECIAL_GPP_UNIT = {"E3SM-1-1-ECA"}
SPECIAL_NPP_UNIT = set()
SPECIAL_RA_UNIT = {"E3SM-1-1", "E3SM-1-1-ECA"}
### !!! 以下模型的原始文件存在已知符号问题 !!!
SPECIAL_NPP_SIGN = {"SAM0-UNICON"}

print(f"Special Models: {(
    SPECIAL_CARBON_UNIT | SPECIAL_GPP_UNIT |
    SPECIAL_NPP_UNIT | SPECIAL_RA_UNIT | SPECIAL_NPP_SIGN
)}")


def find_matching(pattern: str, files: list[Path]) -> Path | None:
    for file in files:
        if file.match(pattern):
            return file
    return None


def model_key_from_cveg(file: Path) -> str:
    """返回 Lmon_<model>_historical，用于匹配同一模型的不同变量。"""
    keywords = file.name.split("_")
    return "_".join(keywords[1:4])


def read_2d(file: Path | None, var_name: str) -> np.ndarray | None:
    """读取已经时间平均和重采样的二维变量，并检查目标网格大小。"""
    if file is None:
        return None
    with warnings.catch_warnings(action="ignore"):
        with xr.open_dataset(file) as ds:
            data = ds[var_name].squeeze(drop=True)
            units = data.attrs.get("units", "unknown")
            val = np.asarray(data, dtype=np.float32)
    if val.shape != RESAMPLED_SHAPE:
        raise ValueError(f"Mismatched shape of {var_name} in {file.name}: {val.shape}")
    print(f"{var_name}: {val.shape}, {units}")
    return val


def read_flux(file: Path | None, var_name: str) -> np.ndarray | None:
    val = read_2d(file, var_name)
    if val is None:  return None
    return (val * FLUX_SCALE_FACTOR).astype(np.float32)


def stat_text(val: np.ndarray) -> str:
    mean = np.nanmean(val)
    std = np.nanstd(val)
    return f"{mean:.4g} ± {std:.4g}"


files = sorted(ROOT_PATH.glob("*Lmon*.nc"))
if not files:
    raise FileNotFoundError(f"{ROOT_PATH} 中没有重采样后的 NetCDF 文件。")

cveg_data = sorted(file for file in files if file.name.startswith("cVeg_Lmon"))
if not cveg_data:
    raise FileNotFoundError(f"{ROOT_PATH} 中没有找到 cVeg 重采样文件。")

model_keys = [model_key_from_cveg(file) for file in cveg_data]
print(f"cVeg models ({len(model_keys)}):")
print([key.split("_")[1] for key in model_keys])


### ---------- cVeg / cRoot / treeFrac 预处理 ---------- ###
print("\n" + "="*30)
print("CMIP6 cVeg/cRoot/treeFrac preprocessing...")
raw_data = OrderedDict()

for idx, (cveg_file, key) in enumerate(zip(cveg_data, model_keys)):
    _, model, _ = key.split("_", 2)
    print(f"\n{idx}: {model}")

    croot_file = find_matching(f"cRoot_{key}*.nc", files)
    tree_file = find_matching(f"treeFrac_{key}*.nc", files)

    # ----- cVeg ----- #
    cveg_val = read_2d(cveg_file, "cVeg")
    cveg_val[cveg_val <= 0] = np.nan
    cveg_val *= 10.0  # kgC m-2 -> MgC ha-1
    if model in SPECIAL_CARBON_UNIT:
        cveg_val /= 1000.0  # fix unit

    # ----- cRoot ----- #
    croot_val = read_2d(croot_file, "cRoot")
    if croot_val is None:
        print(" > No cRoot data")
    else:
        croot_val[croot_val <= 0] = np.nan
        croot_val *= 10.0  # kgC m-2 -> MgC ha-1
        if model in SPECIAL_CARBON_UNIT:
            croot_val /= 1000.0  # fix unit

    # ----- treeFrac ----- #
    treefrac_val = read_2d(tree_file, "treeFrac")
    if treefrac_val is None:
        print(" > No treeFrac data")
    else:
        treefrac_val[treefrac_val <= 0] = np.nan
        treefrac_val /= 100.0  # % -> fraction

    # Save data
    print(f"cVeg data: {stat_text(cveg_val)}")
    raw_data[model] = {
        "cVeg": cveg_val.astype(np.float32),
        "cRoot": None if croot_val is None else croot_val.astype(np.float32),
        "treeFrac": None if treefrac_val is None else treefrac_val.astype(np.float32),
    }

dump(raw_data, RAW_OUT_PATH)
print(f"\nSaved to: {RAW_OUT_PATH}")


### ---------- GPP / NPP / Ra 预处理 ---------- ###
print("\n" + "="*30)
print("Supplementary data preprocessing...")
sup_data = OrderedDict()

for idx, key in enumerate(model_keys):
    _, model, _ = key.split("_", 2)
    print(f"\n{idx}: {model}")

    gpp_file = find_matching(f"gpp_{key}*.nc", files)
    npp_file = find_matching(f"npp_{key}*.nc", files)
    ra_file = find_matching(f"ra_{key}*.nc", files)

    gpp_val = read_flux(gpp_file, "gpp")
    npp_val = read_flux(npp_file, "npp")

    if gpp_val is not None and model in SPECIAL_GPP_UNIT:
        gpp_val /= 1000.0  # fix unit
    if npp_val is not None and model in SPECIAL_NPP_UNIT:
        npp_val /= 1000.0  # fix unit
    if npp_val is not None and model in SPECIAL_NPP_SIGN:
        npp_val *= -1.0    # flip value sign

    # 没有直接提供 NPP 时，使用 NPP = GPP - Ra。
    if npp_val is None and gpp_val is not None:
        ra_val = read_flux(ra_file, "ra")
        if ra_val is not None:
            if model in SPECIAL_RA_UNIT:
                ra_val /= 1000.0  # fix unit
            npp_val = gpp_val - ra_val
            print("* Ra used to calculate NPP")

    if gpp_val is None:  print(" > No GPP data")
    else:  print(f"GPP data: {stat_text(gpp_val)}")

    if npp_val is None:  print(" > No NPP data")
    else:  print(f"NPP data: {stat_text(npp_val)}")

    sup_data[model] = {
        "GPP": None if gpp_val is None else gpp_val.astype(np.float32),
        "NPP": None if npp_val is None else npp_val.astype(np.float32),
    }

dump(sup_data, SUP_OUT_PATH)
print(f"\nSaved to: {SUP_OUT_PATH}")

### ---------- Tree Mask ---------- ###
print("\n" + "="*30)
print("Calculate Tree Mask...")

treefrac_all = np.asarray(
    [val["treeFrac"] for val in raw_data.values() if val["treeFrac"] is not None])
treefrac_all[np.isnan(treefrac_all)] = 0.0  # set NaN to 0
print(treefrac_all.shape)

# MME Tree Fraction (0 ~ 1) -> Tree Mask (0.5 Threshold)
treefrac_mean = np.mean(treefrac_all, axis=0)
TreeMask = (treefrac_mean > 0.5)
print(f"MME Tree Mask: {np.mean(TreeMask):.4f} ± {np.std(TreeMask):.4f}")

img = plt.imshow(treefrac_mean, vmin=0, vmax=1, origin="lower")
plt.colorbar(img, orientation="horizontal", shrink=0.8)
plt.title("MME Tree Fraction")
plt.tight_layout()
plt.show()

TREE_MASK_PATH = "./treemask.joblib"
TREE_FRAC_PATH = "./treefrac.joblib"
dump(TreeMask, TREE_MASK_PATH)
dump(treefrac_mean, TREE_FRAC_PATH)
print(f"\nSaved to: {TREE_MASK_PATH}; {TREE_FRAC_PATH}")


# ----- #
print("\n" + "="*30)
print("Processing Completed!")
print(f"All Models: {len(raw_data)}")
