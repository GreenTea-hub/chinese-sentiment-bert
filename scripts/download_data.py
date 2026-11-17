# -*- coding: utf-8 -*-
"""数据集自动下载脚本（ModelScope 源，SHA256 校验保证四人数据完全一致）。

团队成员在自己的电脑上运行一条命令即可获得与本仓库作者完全相同的
原始数据集，无需手动查找下载：

    python scripts/download_data.py          # 下载缺失/校验失败的文件
    python scripts/download_data.py --force  # 强制重新下载全部文件

数据来源（均来自 ModelScope 魔搭社区，Apache-2.0 协议）：
    - ChnSentiCorp_htl_all        AiNiklaus/ChnSentiCorp
    - weibo_senti_100k train/test swift/weibo_senti_100k
    - SemEval-2014 Task 4         anzhou0798/semeval2014

仅使用 Python 标准库（urllib/hashlib），无需安装第三方包。
"""

import argparse
import hashlib
import os
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW_DIR = os.path.join(ROOT, "data", "raw")

BASE_URL = "https://modelscope.cn/api/v1/datasets/{ns}/repo?Revision=master&FilePath={file}"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# (目标子目录, 数据集 namespace, 远端文件名, 期望 SHA256)
FILES = [
    ("ChnSentiCorp", "AiNiklaus/ChnSentiCorp", "ChnSentiCorp_htl_all.csv",
     "1d414b9c65caf3a836abdb4efded621968502ad3338e1a9d885ae131c86083f0"),
    ("weibo_senti_100k", "swift/weibo_senti_100k", "train.csv",
     "cb4ad963f43e46f015d8143a462b96ef2eb0b7197557c24cf1ce739aba63de89"),
    ("weibo_senti_100k", "swift/weibo_senti_100k", "test.csv",
     "6e0d7c4d977a45d69eb854a44703e20c8f2678f317b43df0f4a8627fd84ef93a"),
    ("semeval2014", "anzhou0798/semeval2014", "Restaurants_Train_v2.xml",
     "223601da1bded6caa4ef9cf91a7007578141ca6d8ed50d5a5c217565f89d2fc5"),
    ("semeval2014", "anzhou0798/semeval2014", "Restaurants_Test_Data_PhaseA.xml",
     "27304ce4b747f5b9ef196d7ce3601d9c48dcb4c64eaf37394b4462af4bcfe139"),
    ("semeval2014", "anzhou0798/semeval2014", "Restaurants_Test_Data_phaseB.xml",
     "3ca8e63fe883bc694e93e062c68c0ddfa670b41f7f91d1dd6896b057faf94953"),
    ("semeval2014", "anzhou0798/semeval2014", "Laptop_Train_v2.xml",
     "efec2ed73b4efb83b47c697114d800149c558118884afa80362a6e975ba7248c"),
    ("semeval2014", "anzhou0798/semeval2014", "Laptops_Test_Data_PhaseA.xml",
     "aa32e8151213c4c102d36e04ecda71a43233c13ee7323d0ebe855ac90ef0dd3b"),
    ("semeval2014", "anzhou0798/semeval2014", "Laptops_Test_Data_phaseB.xml",
     "056367023b4a005d00a355086210d71e995068f6595ba8a289aa4236141fa861"),
]


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def download_one(subdir, ns, fname, expected_sha):
    """下载单个文件并校验 SHA256。已存在且校验通过则跳过。"""
    dest_dir = os.path.join(RAW_DIR, subdir)
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, fname)

    if os.path.exists(dest):
        if sha256_of(dest) == expected_sha:
            print(f"[skip] {fname} 已存在且校验通过")
            return True
        print(f"[redo] {fname} 已存在但校验失败，重新下载")

    url = BASE_URL.format(ns=ns, file=fname)
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    print(f"[get ] {fname} <- {ns}")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
    except Exception as e:
        print(f"[fail] {fname} 下载失败: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        return False

    actual = sha256_of(dest)
    if actual != expected_sha:
        print(f"[fail] {fname} SHA256 不一致（期望 {expected_sha}，实际 {actual}）")
        os.remove(dest)
        return False
    print(f"[ok  ] {fname} 下载完成，校验通过（{os.path.getsize(dest)} bytes）")
    return True


def main():
    parser = argparse.ArgumentParser(description="下载项目所需数据集到 data/raw/")
    parser.add_argument("--force", action="store_true",
                        help="强制重新下载所有文件（默认跳过已校验的文件）")
    args = parser.parse_args()

    print(f"数据目录: {RAW_DIR}")
    ok = 0
    fail = 0
    for subdir, ns, fname, sha in FILES:
        if args.force and os.path.exists(os.path.join(RAW_DIR, subdir, fname)):
            os.remove(os.path.join(RAW_DIR, subdir, fname))
        if download_one(subdir, ns, fname, sha):
            ok += 1
        else:
            fail += 1
    print(f"\n完成：成功 {ok} 个，失败 {fail} 个")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
