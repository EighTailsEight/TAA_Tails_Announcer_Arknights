r"""语音包：一组 wav + manifest 打成一个 zip，运行时解包到缓存目录再播。

包格式（zip，推荐 STORED 不压缩）：
  manifest.json  {"format":1,"name":..,"author":..,"notes":..,"keys":{key:[wav,..]}}
  *.wav

没有 manifest 也能读：按文件名归组（sell.wav / sell_2.wav / sell_3.wav）。
包内多一条少一条都无所谓：Player 只会取它需要的 key，缺的直接静默跳过。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

FORMAT = 1
MANIFEST = "manifest.json"
VARIANT = re.compile(r"^(.*)_(\d+)$")


def index_names(names) -> dict[str, list[str]]:
    """把一组文件名归到 key：sell.wav / sell_2.wav -> {"sell": [...]}。"""
    stems = sorted({Path(n).stem for n in names if str(n).lower().endswith(".wav")})
    present = set(stems)
    index = {}
    for stem in stems:
        m = VARIANT.match(stem)
        if m and m.group(1) in present:
            continue                      # 变体文件，归到主 key 下
        index[stem] = [stem + ".wav"]
    for stem in stems:
        m = VARIANT.match(stem)
        if m and m.group(1) in present:
            index[m.group(1)].append(stem + ".wav")
    return index


def safe_member(name: str) -> bool:
    """包内条目只能是本目录下的纯文件名，挡掉路径穿越和盘符。"""
    if not name or name in (".", ".."):
        return False
    if ":" in name or "\\" in name or "/" in name:
        return False
    return Path(name).name == name


def read_manifest(path: Path) -> dict:
    """读包信息；manifest 缺失或损坏时按文件名推断 key。"""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        meta: dict = {}
        keys = None
        if MANIFEST in names:
            try:
                raw = json.loads(z.read(MANIFEST).decode("utf-8-sig"))
            except (ValueError, UnicodeDecodeError, KeyError):
                raw = None
            if isinstance(raw, dict):
                meta = raw
                table = raw.get("keys")
                if isinstance(table, dict):
                    keys = {}
                    for key, value in table.items():
                        members = [value] if isinstance(value, str) else value
                        if not isinstance(members, list):
                            continue
                        found = [m for m in members
                                 if isinstance(m, str) and m in names
                                 and safe_member(m)]
                        if found:
                            keys[key] = found
        if keys is None:
            keys = index_names(names)
        return {
            "name": str(meta.get("name") or path.stem),
            "author": str(meta.get("author") or ""),
            "notes": str(meta.get("notes") or ""),
            "keys": keys,
        }


def summarize(path: Path) -> dict:
    """给 GUI/工具用：包名、key 数、文件数。"""
    info = read_manifest(path)
    files = {m for members in info["keys"].values() for m in members}
    info["files"] = len(files)
    info["size"] = Path(path).stat().st_size
    return info


def pack_tag(path: Path) -> str:
    st = Path(path).stat()
    return f"{Path(path).stem}-{st.st_mtime_ns:x}-{st.st_size:x}"


def extract(path: Path, dest: Path) -> dict[str, list[Path]]:
    """解包到 dest（已存在且非空的条目直接复用），返回 key -> 文件路径。"""
    info = read_manifest(path)
    dest.mkdir(parents=True, exist_ok=True)
    index: dict[str, list[Path]] = {}
    with zipfile.ZipFile(path) as z:
        for key, members in info["keys"].items():
            paths = []
            for member in members:
                target = dest / member
                if not target.is_file() or target.stat().st_size == 0:
                    with z.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 256 * 1024)
                paths.append(target)
            index[key] = paths
    return index


def build(src: Path, out: Path, name: str, author: str = "",
          notes: str = "", compress: bool = False) -> dict:
    """把 src 目录下的 wav 打进 out；返回写进包里的 manifest。"""
    src = Path(src)
    files = sorted(p for p in src.glob("*.wav") if p.is_file())
    if not files:
        raise ValueError(f"{src} 里没有 wav 文件")
    keys = index_names(p.name for p in files)
    manifest = {
        "format": FORMAT,
        "name": name or src.name,
        "author": author,
        "notes": notes,
        "keys": keys,
    }
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(tmp, "w", mode) as z:
        z.writestr(MANIFEST, json.dumps(manifest, ensure_ascii=False, indent=2))
        for p in files:
            z.write(p, p.name)
    os.replace(tmp, out)
    return manifest
