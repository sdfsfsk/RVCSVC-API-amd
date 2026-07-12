# RVCSVC-API-amd

面向 [astrbot_plugin_matsuko_cover](https://github.com/sdfsfsk/matsuko_cover) 的 RVC / SVC-Fusion 中间层，使用 UVR5 分离人声和伴奏，并提供 Gradio API、结果缓存、音频后处理和进度回传。

> [!CAUTION]
> **此版本仅面向 Windows AMD 显卡（A 卡），使用 DirectML。NVIDIA、Intel GPU 和纯 CPU 环境不在支持范围内。**

本仓库仅发布源码和环境安装脚本，不包含便携 Python、UVR5 权重、RVC/SVC 模型、歌曲、缓存或生成音频。

## 与 MSST 版本的区别

| 项目 | RVCSVC-API-amd | [RVCSVC-API-MSST](https://github.com/sdfsfsk/RVCSVC-API-MSST) |
|---|---|---|
| 分离器 | UVR5 / HP5 | BS-Roformer / MSST |
| GPU 后端 | DirectML | Windows AMD ROCm 7.2.1 |
| 特点 | 环境较轻、兼容范围较广 | 分离质量更高、显存占用更大 |
| 支持显卡 | 仅 AMD | 仅 AMD |

两套中间层使用相同的 3333/9999 端口，不能同时启动。

## 数据流和端口

```text
AstrBot + matsuko_cover
  ├─ RVC 请求 → RVCSVC-API-amd :3333 → UVR5 → RVC :2333
  └─ SVC 请求 → RVCSVC-API-amd :9999 → UVR5 → SVC-Fusion :7777
```

## 安装

要求：

- Windows 10/11 x64
- AMD Radeon GPU 和可用的 DirectML 驱动
- FFmpeg 已加入 `PATH`
- 已准备上游 RVC 或 SVC-Fusion 服务

运行：

```text
install_env.bat
```

脚本会下载官方 CPython 3.10.11 Embedded 到 `Python310/`，并安装 `requirements.txt`。也可以使用现有 Python 3.10 环境手动安装：

```powershell
python -m pip install -r requirements.txt
```

## UVR5 模型

从 [Ultimate Vocal Remover GUI](https://github.com/Anjok07/ultimatevocalremovergui) 项目认可的模型来源获取 `5_HP-Karaoke-UVR.pth`，放到：

```text
uvr5/uvr_model/5_HP-Karaoke-UVR.pth
```

模型文件不在本仓库中分发。

## 启动

先启动上游推理服务，再启动需要的中间层：

```text
启动 rvcapi.bat   # 中间层 3333，需要 RVC 2333
启动 svcapi.bat   # 中间层 9999，需要 SVC-Fusion 7777
```

手动启动：

```powershell
Python310\python.exe app_rvc.py --dml --is_nohalf
Python310\python.exe app_svc.py --dml --is_nohalf
```

插件默认地址：

```text
rvc_base_url = http://127.0.0.1:3333/
svc_base_url = http://127.0.0.1:9999/
```

## API

- `/convert`：下载/读取歌曲、UVR5 分离、调用上游、混音并返回结果。
- `/show_model`：读取上游可用模型。
- 结果按歌曲和所有关键参数哈希保存在 `temp/`；参数完全一致时直接返回缓存。
- `app.queue(..., api_open=True)` 必须保持启用，否则 AstrBot 的 `gradio_client` 无法调用。

## 注意事项

- 服务会调用本机上游端口，不要将 3333/9999 直接暴露到公网。
- `output/`、`temp/` 和下载音频可能包含受版权保护内容，已默认被 Git 忽略。
- DirectML 不支持所有 CUDA 专用算子；若需要更高分离质量，使用 MSST 的原生 AMD ROCm 版本。
- 本项目不会自动提供上游 RVC/SVC 模型，使用者需自行遵守模型和音频许可。

## 来源与许可状态

中间层基于 [CCYellowStar2/RVCSVC-API](https://github.com/CCYellowStar2/RVCSVC-API) 修改；该上游仓库在本项目发布时未声明开源许可证，因此本仓库不擅自为继承代码授予额外许可。第三方 UVR5 组件及其许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。

