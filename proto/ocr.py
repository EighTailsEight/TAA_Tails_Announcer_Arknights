r"""金币数字 OCR：PP-OCRv6 rec small（21 MB），只识别、不检测。

read_gold 的 14×22 字形模板只覆盖 0/1/4/5/6/9；数字闪光/变形时模板会漏读。
这里直接喂 onnxruntime，省掉 RapidOCR/cv2 整条检测链和 ~150 MB 依赖：
crop → 4× LANCZOS → 高 48 定宽 320 填充 → CTC 贪心解码。

任何一步不可用（没装 onnxruntime、模型缺失）都静默返回 None，模板读数仍是主路径。
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
from PIL import Image

from screen_detector import crop_canonical

MODEL = Path(__file__).resolve().parent / "models" / "PP-OCRv6_rec_small.onnx"
GOLD_OCR_BOX = (1778, 798, 1844, 852)   # 1920 规范坐标：金币数字紧框
REC_H = 48
REC_MAX_W = 320
MIN_SCORE = 0.90                        # 实测金币数字 0.97+；低于此多半是背景纹理
TEXT_MIN_SCORE = 0.60                   # 固定短框实测 0.94–1.00；低于此当噪声
DIGITS = frozenset("0123456789")

_sess = None
_chars = None
_failed = False
_lock = threading.Lock()


def _load() -> bool:
    global _sess, _chars, _failed
    if _sess is not None:
        return True
    if _failed:
        return False
    with _lock:
        if _sess is not None:
            return True
        if _failed:
            return False
        try:
            import onnxruntime as ort
            opt = ort.SessionOptions()
            opt.log_severity_level = 4
            sess = ort.InferenceSession(str(MODEL), sess_options=opt,
                                        providers=["CPUExecutionProvider"])
            # CTC：0 是 blank，字典在末尾补一个空格（与 PaddleOCR 解码一致）
            chars = sess.get_modelmeta().custom_metadata_map["character"].split("\n")
            _chars = ["blank"] + chars + [" "]
            _sess = sess
            return True
        except Exception:
            _failed = True
            return False


def _prep(patch: np.ndarray) -> np.ndarray:
    im = Image.fromarray(patch).resize((patch.shape[1] * 4, patch.shape[0] * 4),
                                       Image.LANCZOS)
    tw = min(int(np.ceil(REC_H * im.width / im.height)), REC_MAX_W)
    im = im.resize((tw, REC_H), Image.BILINEAR)
    x = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
    x = (x - 0.5) / 0.5
    out = np.zeros((3, REC_H, REC_MAX_W), dtype=np.float32)
    out[:, :, :tw] = x
    return out[None]


def _infer(patch: np.ndarray) -> tuple[str, float]:
    out = _sess.run(None, {_sess.get_inputs()[0].name: _prep(patch)})[0][0]
    idx = out.argmax(axis=1)
    prob = out.max(axis=1)
    keep = np.ones(len(idx), dtype=bool)
    keep[1:] = idx[1:] != idx[:-1]
    keep &= idx != 0
    if not keep.any():
        return "", 0.0
    text = "".join(_chars[int(i)] for i in idx[keep])
    return text, float(prob[keep].mean())


def warmup() -> bool:
    """提前把会话建好并空跑一次（导入 onnxruntime 0.2s + 首帧 0.3s），
    实时监控开跑时放后台线程调用，免得第一次识别主界面文字白等半秒。"""
    if not _load():
        return False
    try:
        _infer(np.full((8, 32, 3), 255, dtype=np.uint8))
    except Exception:
        return False
    return True


def read_patch_text(patch: np.ndarray, min_score: float = TEXT_MIN_SCORE) -> str | None:
    """已裁好的补丁直接识别（裁片做过掩码/放大的场景用）。"""
    if not _load():
        return None
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    text, score = _infer(patch)
    if not text or score < min_score:
        return None
    return text


def read_text(rgb, box, min_score: float = TEXT_MIN_SCORE) -> str | None:
    """固定 ROI 上的通用文字读数（不做数字过滤）。框要贴紧一行字，rec 不做检测。"""
    if not _load():
        return None
    patch = crop_canonical(rgb, box)
    return read_patch_text(patch, min_score)


def read_gold_ocr(rgb) -> int | None:
    """金币读数；模板读不出时兜底。非纯数字或置信度不足一律 None。"""
    if not _load():
        return None
    patch = crop_canonical(rgb, GOLD_OCR_BOX)
    if patch.size == 0 or min(patch.shape[:2]) < 8:
        return None
    text, score = _infer(patch)
    if not text or any(c not in DIGITS for c in text):
        return None
    if score < MIN_SCORE:
        return None
    return int(text)
