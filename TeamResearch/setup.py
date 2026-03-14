"""
TeamResearch 数据集下载脚本
运行方式：python setup.py
会自动下载并解压项目所需的三个数据集到 Dataset/ 目录。
"""

import io
import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

DATASETS = [
    {
        "name": "UCI HAR (人体活动识别)",
        "url": "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip",
        "dest": DATASET_DIR / "human+activity+recognition+using+smartphones",
    },
    {
        "name": "PAMAP2 (体力活动监测)",
        "url": "https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip",
        "dest": DATASET_DIR / "PAMAP2",
    },
    {
        "name": "WESAD Kaggle (可穿戴压力检测)",
        "url": "kaggle://teejmahal20/wearable-stress-detect",
        "dest": DATASET_DIR / "WESAD_Kaggle",
    },
]


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_done = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  下载中... {mb_done:.1f}/{mb_total:.1f} MB ({percent:.0f}%)")
    else:
        mb_done = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  下载中... {mb_done:.1f} MB")
    sys.stdout.flush()


def download_and_extract_zip(url, dest):
    """下载 zip 文件并解压到目标目录"""
    dest.mkdir(parents=True, exist_ok=True)
    tmp_path = dest / "_download_tmp.zip"
    try:
        urllib.request.urlretrieve(url, str(tmp_path), reporthook=_progress_hook)
        print()
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"下载失败: {e}")

    print("  解压中...")
    with zipfile.ZipFile(str(tmp_path), "r") as zf:
        zf.extractall(str(dest))
    tmp_path.unlink(missing_ok=True)


def download_kaggle_dataset(dataset_slug, dest):
    """通过 kaggle API 下载数据集"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("  [!] 需要安装 kaggle 包: pip install kaggle")
        print("  [!] 并配置 ~/.kaggle/kaggle.json (Kaggle API Token)")
        print(f"  [!] 然后手动运行: kaggle datasets download -d {dataset_slug} -p \"{dest}\" --unzip")
        return False

    dest.mkdir(parents=True, exist_ok=True)
    print("  通过 Kaggle API 下载中...")
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(dest), unzip=True)
    return True


def check_dataset_exists(dest):
    """检查数据集目录是否已存在且非空"""
    if not dest.exists():
        return False
    contents = list(dest.iterdir())
    return len(contents) > 0


def main():
    print("=" * 58)
    print("   TeamResearch 数据集下载工具")
    print("=" * 58)
    print(f"数据集目录: {DATASET_DIR}\n")

    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for i, ds in enumerate(DATASETS, 1):
        print(f"[{i}/{len(DATASETS)}] {ds['name']}")

        if check_dataset_exists(ds["dest"]):
            print(f"  已存在，跳过: {ds['dest']}\n")
            continue

        url = ds["url"]
        try:
            if url.startswith("kaggle://"):
                slug = url.replace("kaggle://", "")
                success = download_kaggle_dataset(slug, ds["dest"])
                if not success:
                    all_ok = False
            else:
                download_and_extract_zip(url, ds["dest"])
            print(f"  完成: {ds['dest']}\n")
        except Exception as e:
            print(f"  失败: {e}\n")
            all_ok = False

    print("=" * 58)
    if all_ok:
        print("所有数据集准备就绪!")
    else:
        print("部分数据集需要手动处理，请查看上方提示。")
    print("=" * 58)

    # 提示后续步骤
    print("\n后续步骤:")
    print("  1. cd fall_detection_backend")
    print("  2. pip install -r requirements.txt")
    print("  3. 复制 .env.example 为 .env 并填入 QWEN_API_KEY")
    print("  4. python main.py")
    print("  5. 浏览器访问 http://localhost:8000")


if __name__ == "__main__":
    main()
