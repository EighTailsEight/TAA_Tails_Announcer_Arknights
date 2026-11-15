# 语音包目录

把 `*.zip` 语音包放在这里，程序启动时会自动扫描并加入下拉框。

仓库不附带语音包。没有语音包时程序照常运行，只是不发声。

## 打包

```bat
python tools\voice_pack_tool.py --src <wav目录> --out packs\mypack.zip --name 我的语音包
```

或者双击 `tools\voice_pack_tool.bat` 使用图形界面。

## 文件命名

- `battle_enter.wav`：key = `battle_enter`
- `battle_enter_2.wav`、`battle_enter_3.wav`：同一 key 的随机变体，每次随机播放一条

完整 key 列表与语义见 [docs/KEYS.md](../docs/KEYS.md)。包里缺少的 key 会被静默跳过，不会报错。

`packs/*.zip` 已被 `.gitignore` 忽略，不会被提交。
