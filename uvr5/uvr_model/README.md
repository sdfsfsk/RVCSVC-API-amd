# UVR5 模型下载说明

本目录需要以下人声分离模型。模型权重体积较大，不随本仓库分发。

| 文件名 | 下载源 | 大小 | SHA-256 |
| --- | --- | ---: | --- |
| `5_HP-Karaoke-UVR.pth` | [UVR 公共模型 GitHub Release](https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/5_HP-Karaoke-UVR.pth) | 126,782,699 字节 | `FE00891DEFBB61F4261500AF22F7624F1A3DF8DC75FA3998D1AECE02E6BE4537` |

该文件也收录在 [Ultimate Vocal Remover GUI 官方手动下载目录](https://github.com/Anjok07/ultimatevocalremovergui/blob/master/gui_data/model_manual_download.json) 中。

## 一键下载

在本目录打开 PowerShell，执行：

```powershell
curl.exe -L "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/5_HP-Karaoke-UVR.pth" -o "5_HP-Karaoke-UVR.pth"
```

下载完成后必须保持文件名为 `5_HP-Karaoke-UVR.pth`，并放在当前目录：

```text
uvr5/uvr_model/5_HP-Karaoke-UVR.pth
```

## 校验文件

```powershell
Get-FileHash -Algorithm SHA256 ".\5_HP-Karaoke-UVR.pth"
```

输出哈希应与上表一致。若不一致，请删除文件后重新下载。
