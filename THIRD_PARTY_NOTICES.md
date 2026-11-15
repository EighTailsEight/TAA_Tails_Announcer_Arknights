# 第三方组件与素材

## 运行时依赖

| 组件 | 版本 | 许可 | 说明 |
|---|---|---|---|
| numpy | >=1.24 | BSD-3-Clause | 数组与模板匹配 |
| Pillow | >=10.0 | HPND | 图像缩放、WAV 无关的图像处理 |
| onnxruntime | 1.24.4 (CPU) | MIT | 运行 OCR 模型 |
| PyInstaller | 6.22.3（仅构建） | GPL-2.0 + bootloader exception | 打包 exe；例外条款允许分发打包产物 |
| Python / Tk | 3.12+ | PSF License | 运行时与 GUI |

## 模型

`proto/models/PP-OCRv6_rec_small.onnx` 为 PaddleOCR 的 PP-OCRv6 rec small 文字识别模型转换的 ONNX 文件。PaddleOCR 以 Apache-2.0 许可发布；再分发时请保留其许可证与版权声明，并确认所用版本的具体条款。

## 游戏素材

`proto/templates/` 下是游戏画面中裁出的少量 UI 区域（按钮、徽章、血条、结算页文字等），仅用于状态识别。相关美术与商标版权归鹰角网络所有，本项目与其无关联。若权利方提出异议，会移除这些素材并改为由用户自行采集。

## 语音

仓库不包含任何语音文件。语音包由用户自行制作和分发，其版权与使用许可由制作者负责。
