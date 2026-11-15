r"""语音包打包工具。

双击（voice_pack_tool.bat）开窗口选文件夹、填包名、生成 .zip；
也可以命令行批量打包：

  python voice_pack_tool.py --src ..\voices --out ..\packs\muyin.zip --name 谬因语音包
  python voice_pack_tool.py --list ..\packs\muyin.zip
  python voice_pack_tool.py --unpack ..\packs\muyin.zip --out ..\voices_unpacked

包是普通 zip：manifest.json + *.wav。程序只认它需要的 key，缺的、多的都不会报错；
文件名带 _2 / _3 后缀的算同一 key 的随机变体（sell.wav / sell_2.wav / sell_3.wav）。
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / "app"))

import voice_pack  # noqa: E402


def known_keys() -> set[str]:
    try:
        from player import known_keys as keys
        return keys()
    except Exception:
        return set()


def scan(src: Path) -> tuple[dict[str, list[str]], list[str]]:
    files = sorted(p for p in Path(src).glob("*.wav") if p.is_file())
    index = voice_pack.index_names(p.name for p in files)
    known = known_keys()
    unknown = sorted(k for k in index if known and k not in known)
    return index, unknown


def do_build(src: Path, out: Path, name: str, author: str = "", notes: str = "",
             compress: bool = False, log=print) -> dict:
    index, unknown = scan(src)
    if unknown:
        log(f"提示：{len(unknown)} 个文件名程序不会用到（可能拼错）："
            + "、".join(unknown[:8]) + ("…" if len(unknown) > 8 else ""))
    manifest = voice_pack.build(src, out, name, author, notes, compress)
    size = out.stat().st_size
    log(f"已生成 {out}（{len(manifest['keys'])} 个 key / "
        f"{size / 1e6:.1f} MB）")
    return manifest


def do_list(path: Path, log=print) -> dict:
    info = voice_pack.summarize(path)
    log(f"{path}\n  名称：{info['name']}  作者：{info['author'] or '—'}")
    log(f"  {len(info['keys'])} 个 key / {info['files']} 个文件 / "
        f"{info['size'] / 1e6:.1f} MB")
    if info["notes"]:
        log(f"  备注：{info['notes']}")
    missing = sorted(known_keys() - set(info["keys"]))
    if missing:
        log(f"  包里没有的 key（程序会静默跳过）：{'、'.join(missing)}")
    return info


def do_unpack(path: Path, out: Path, log=print) -> dict:
    index = voice_pack.extract(path, out)
    log(f"已解包到 {out}（{sum(len(v) for v in index.values())} 个文件）")
    return index


class App:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import filedialog, messagebox, ttk

        self.tk = tk
        self.filedialog = filedialog
        self.messagebox = messagebox
        root.title("Tails Announcer 语音包打包工具")
        root.geometry("760x560")
        root.minsize(640, 480)

        pad = {"padx": 8, "pady": 5}
        frame = ttk.Frame(root, padding=10)
        frame.pack(fill="both", expand=True)

        self.src = tk.StringVar()
        self.name = tk.StringVar()
        self.author = tk.StringVar()
        self.out = tk.StringVar()

        rows = ttk.Frame(frame)
        rows.pack(fill="x")
        ttk.Label(rows, text="语音文件夹").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(rows, textvariable=self.src).grid(row=0, column=1, columnspan=2,
                                                    sticky="we", **pad)
        ttk.Button(rows, text="浏览", command=self.pick_src).grid(row=0, column=3, **pad)
        ttk.Label(rows, text="包名").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(rows, textvariable=self.name).grid(row=1, column=1, sticky="we", **pad)
        ttk.Label(rows, text="作者").grid(row=1, column=2, sticky="e", **pad)
        ttk.Entry(rows, textvariable=self.author).grid(row=1, column=3, sticky="we", **pad)
        ttk.Label(rows, text="输出文件").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(rows, textvariable=self.out).grid(row=2, column=1, columnspan=2,
                                                    sticky="we", **pad)
        ttk.Button(rows, text="另存为", command=self.pick_out).grid(row=2, column=3, **pad)
        rows.columnconfigure(1, weight=1)

        self.summary = tk.StringVar(value="选择文件夹后会自动统计")
        ttk.Label(frame, textvariable=self.summary).pack(fill="x", **pad)

        bar = ttk.Frame(frame)
        bar.pack(fill="x", **pad)
        ttk.Button(bar, text="重新扫描", command=self.preview).pack(side="left")
        ttk.Button(bar, text="打包", command=self.build).pack(side="left", padx=8)
        ttk.Button(bar, text="查看现有包", command=self.list_pack).pack(side="left")
        ttk.Button(bar, text="解包到文件夹", command=self.unpack).pack(side="left", padx=8)

        self.log = tk.Text(frame, height=16, wrap="word")
        self.log.pack(fill="both", expand=True, **pad)
        self.write = lambda msg: (self.log.insert("end", str(msg) + "\n"),
                                  self.log.see("end"))

        self.src.trace_add("write", lambda *_: self.preview())

    def pick_src(self):
        d = self.filedialog.askdirectory(title="选择放 wav 的文件夹")
        if not d:
            return
        self.src.set(d)
        if not self.name.get():
            self.name.set(Path(d).name)
        if not self.out.get():
            self.out.set(str(BASE / "packs" / (Path(d).name + ".zip")))

    def pick_out(self):
        f = self.filedialog.asksaveasfilename(
            title="保存语音包", defaultextension=".zip",
            filetypes=[("语音包", "*.zip")])
        if f:
            self.out.set(f)

    def preview(self):
        src = Path(self.src.get())
        if not src.is_dir():
            self.summary.set("选择文件夹后会自动统计")
            return
        index, unknown = scan(src)
        files = sum(len(v) for v in index.values())
        text = f"{files} 个 wav / {len(index)} 个 key"
        if unknown:
            text += f" · 未识别的文件名 {len(unknown)} 个：" + "、".join(unknown[:5])
        self.summary.set(text)

    def build(self):
        src = Path(self.src.get())
        out = Path(self.out.get() or (BASE / "packs" / (self.name.get() + ".zip")))
        if not src.is_dir():
            self.messagebox.showwarning("缺少文件夹", "先选一个放 wav 的文件夹")
            return
        if not self.name.get().strip():
            self.messagebox.showwarning("缺少包名", "给语音包起个名字")
            return
        try:
            manifest = do_build(src, out, self.name.get().strip(),
                                self.author.get().strip(), log=self.write)
        except Exception as exc:
            self.messagebox.showerror("打包失败", repr(exc))
            return
        self.write(f"完成：{len(manifest['keys'])} 个 key")
        self.messagebox.showinfo("打包完成", f"已生成\n{out}")

    def list_pack(self):
        f = self.filedialog.askopenfilename(
            title="选择语音包", filetypes=[("语音包", "*.zip")])
        if f:
            try:
                do_list(Path(f), log=self.write)
            except Exception as exc:
                self.messagebox.showerror("读取失败", repr(exc))

    def unpack(self):
        f = self.filedialog.askopenfilename(
            title="选择语音包", filetypes=[("语音包", "*.zip")])
        if not f:
            return
        d = self.filedialog.askdirectory(title="解包到哪个文件夹")
        if not d:
            return
        try:
            do_unpack(Path(f), Path(d), log=self.write)
        except Exception as exc:
            self.messagebox.showerror("解包失败", repr(exc))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="语音包打包工具")
    parser.add_argument("--src", type=Path, help="放 wav 的文件夹")
    parser.add_argument("--out", type=Path, help="输出 zip 路径")
    parser.add_argument("--name", default="", help="包名")
    parser.add_argument("--author", default="", help="作者")
    parser.add_argument("--notes", default="", help="备注")
    parser.add_argument("--compress", action="store_true", help="deflate 压缩（默认不压缩）")
    parser.add_argument("--list", type=Path, dest="list_pack", help="查看包信息")
    parser.add_argument("--unpack", type=Path, help="解包 zip")
    args = parser.parse_args(argv)

    if args.unpack:
        do_unpack(args.unpack, args.out or (args.unpack.parent / args.unpack.stem))
        return 0
    if args.list_pack:
        do_list(args.list_pack)
        return 0
    if args.src:
        if not args.out:
            args.out = BASE / "packs" / ((args.name or args.src.name) + ".zip")
        do_build(args.src, args.out, args.name or args.src.name,
                 args.author, args.notes, args.compress)
        return 0

    import tkinter as tk
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
