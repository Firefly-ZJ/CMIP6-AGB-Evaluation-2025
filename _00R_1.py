#####     CMIP6 Temporal Mean and Resampling     #####
# 对原始 CMIP6 月尺度数据计算 2010~2014 年平均，并用 CDO 重采样到 1° 网格。
# 本脚本只进行时间平均和空间重采样，不进行单位转换、缺测值处理等预处理。
from pathlib import Path
import os
import re
import shutil
import subprocess


SOURCE_DIRS = (Path("./Historical/"), Path("./Historical-2/"))
OUT_DIR = Path("./Data/Resampled/")

START_DATE = "2010-01-01"
END_DATE = "2014-12-31"
TARGET_GRID = "r360x180"  # 全球规则经纬度网格，分辨率为 1° × 1°
CDO_THREADS = 4


def cdo_prefix() -> tuple[list[str], bool]:
    """返回 CDO 命令前缀，以及是否需要将路径转换为 WSL 格式。"""
    if shutil.which("cdo") is not None:
        return ["cdo"], False
    if os.name == "nt" and shutil.which("wsl") is not None:
        return ["wsl", "-e", "cdo"], True
    raise RuntimeError(
        "未找到 CDO。请在 Linux/WSL 中安装 cdo，或确保 cdo 已加入 PATH。"
    )


def wsl_path(path: Path) -> str:
    """把绝对 Windows 路径转换为 WSL 中的 /mnt/<drive>/... 路径。"""
    path = path.resolve()
    drive = path.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"无法转换为 WSL 路径: {path}")
    relative = path.as_posix()[len(path.drive):].lstrip("/")
    return f"/mnt/{drive}/{relative}"


def output_name(source: Path) -> str:
    """用实际处理时段和目标分辨率替换原文件名中的完整历史时段。"""
    stem = re.sub(r"_\d{6}-\d{6}$", "_201001-201412_1deg", source.stem)
    if stem == source.stem:
        stem += "_201001-201412_1deg"
    return stem + source.suffix


def run_cdo(source: Path, output: Path, prefix: list[str], use_wsl: bool):
    """先计算指定时段平均，再进行一阶守恒重采样。"""
    source_arg = wsl_path(source) if use_wsl else str(source.resolve())
    output_arg = wsl_path(output) if use_wsl else str(output.resolve())

    # CDO 从右向左执行操作符：seldate -> timmean -> remapcon。
    command = prefix + [
        "-O",
        "-P", str(CDO_THREADS),
        "-f", "nc4",
        f"remapcon,{TARGET_GRID}",      # 一阶守恒重采样
        "-timmean",                     # 时间平均
        f"-seldate,{START_DATE},{END_DATE}",  # 选择指定时段
        source_arg,
        output_arg,
    ]
    subprocess.run(command, check=True)


print("CMIP6 temporal averaging and conservative resampling...")
OUT_DIR.mkdir(parents=True, exist_ok=True)

prefix, use_wsl = cdo_prefix()
files = sorted(
    file
    for source_dir in SOURCE_DIRS
    for file in source_dir.glob("*Lmon*.nc")
)

if not files:
    raise FileNotFoundError("输入目录中没有找到 Lmon NetCDF 文件。")
else:
    print(f"Total NetCDF files: {len(files)}")

for idx, source in enumerate(files, start=1):
    output = OUT_DIR / output_name(source)
    print(f"[{idx}/{len(files)}] {source.name}")
    print(f"  -> {output}")
    run_cdo(source, output, prefix, use_wsl)

print("\n" + "="*30)
print("Processing Completed!")
print(f"Processed files: {len(files)},  Saved to: {OUT_DIR}")
