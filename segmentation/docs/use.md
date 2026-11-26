# SegMAN 在 Windows 上训练与测试 General 光伏数据集

本指南使用 Windows 64 位、PowerShell 和单张 NVIDIA GPU。训练使用项目的 [tools/train.py](../tools/train.py)，测试使用 [tools/test.py](../tools/test.py)。[General 配置](../local_configs/segman/tiny/segman_t_general.py) 继承 SegMAN-T 的 ADE20K 配置，使用 General 的 train、val、test 和两类标签。

Windows 下，Triton、NATTEN 和 `kernels/selective_scan` 对应的算子由 PyTorch 在 GPU 上执行。GPU 依赖使用 Windows 预编译 wheel。PyTorch 算子版训练和推理的速度可能低于原项目的专用 CUDA 内核。

## 1. 创建环境（任意目录）

打开 **Anaconda PowerShell Prompt**，确认 NVIDIA 驱动能识别 GPU，再创建并激活环境。后续安装、训练和测试都在激活后的终端执行。

```powershell
nvidia-smi
conda create -n segman python=3.11 pip -y
conda activate segman
pip --version
```

`pip --version` 输出的路径应指向 `segman`。激活后可直接使用 `pip`。

## 2. 安装 Python 依赖（任意目录）

这些包从软件源或 wheel 索引安装，可以在当前目录直接执行。CUDA 11.8 运行时随 PyTorch wheel 安装。对应版本见 [PyTorch 2.0.1](https://pytorch.org/get-started/previous-versions/#v201) 和 [MMCV Windows wheel 索引](https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html)。

```powershell
pip install setuptools==69.5.1 "wheel<0.46"
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 opencv-python==4.10.0.84 pillow==10.4.0 yapf==0.40.2 matplotlib==3.9.2 prettytable==3.11.0
pip install timm==0.9.16 einops==0.8.0 fvcore==0.1.5.post20221221 tensorboard==2.17.1
pip install mmcv-full==1.7.2 --only-binary mmcv-full --find-links https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
```

MMCV 1.7.2 的 Windows wheel 依赖 `opencv-python`。上述命令先安装 4.10.0.84，便于保留与 PyTorch 2.0.1 兼容的 NumPy 1.26.4。如果旧安装出现 `RuntimeError: Numpy is not available`，在 `segman` 环境中恢复这组版本。

```powershell
conda activate segman
pip uninstall -y opencv-python-headless
pip install --force-reinstall numpy==1.26.4 opencv-python==4.10.0.84
pip check
```

## 3. 安装项目并检查 GPU（必须在 segmentation 目录）

`pip install -e .` 中的点表示当前目录。先进入包含 `setup.py` 的 `segmentation`，再安装仓库内的 MMSegmentation。

```powershell
cd C:\ZZworkspace\zyworkspace\project\codes\dis\SegMAN\segmentation
pip install --no-build-isolation -e .
pip check
python -c "import mmcv, mmcv.ops, mmseg, numpy as np, torch; from mmseg.models import build_segmentor; assert torch.cuda.is_available(), 'CUDA unavailable'; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0)); print('MMCV', mmcv.__version__, 'MMSeg', mmseg.__file__); print('NumPy', np.__version__, torch.from_numpy(np.zeros(1, dtype=np.float32)))"
python tools/train.py --help
python tools/test.py --help
```

检查应显示 PyTorch `2.0.1+cu118`、CUDA `11.8`、GPU 名称、MMCV `1.7.2` 和仓库内的 `mmseg` 路径。

## 4. 数据与训练（segmentation 目录）

数据根目录为 `C:\ZZworkspace\zyworkspace\project\datasets\dis\General`，直接写在配置的 `data_root` 中。每组下的 `image` 和 `label` 使用同名 `.tif`。影像为 RGB，单通道标签中 `0` 是背景，`1` 是光伏面板。

| 用途 | 图像目录 | 标签目录 | 样本数 |
| --- | --- | --- | ---: |
| 训练 | train/image | train/label | 17,648 |
| 验证 | val/image | val/label | 1,878 |
| 最终测试 | test/image | test/label | 4,802 |

默认输入 512 × 512，与 General 原始 TIFF 尺寸一致。单卡每批处理 8 张图，每轮训练 2,206 步，共训练 50 轮。每轮结束后验证一次并记录四联图。下面的命令使用 ImageNet-1K 权重初始化 SegMAN-T 编码器，再在 General 上训练。

```powershell
cd C:\ZZworkspace\zyworkspace\project\codes\dis\SegMAN\segmentation
python tools/train.py local_configs/segman/tiny/segman_t_general.py --gpu-id 0 --seed 42
```

预训练文件路径由配置中的 `model.backbone.pretrained` 指定，当前为 `C:\ZZworkspace\zyworkspace\project\weights\cv\SegMAN\SegMAN_Encoder_t.pth.tar`。训练入口直接读取该文件。[官方编码器权重](https://drive.google.com/drive/folders/1QYU7nhpe0ddH7bPxI7VH4drc__07uEHs?usp=sharing)

在 [General 配置](../local_configs/segman/tiny/segman_t_general.py) 中调整训练参数。

| 用途 | 配置项 | 当前值 |
| --- | --- | --- |
| 训练轮数 | `runner.max_epochs` | 50 |
| 单卡训练 batch size | `data.samples_per_gpu` | 8 |
| 每张 GPU 的默认 DataLoader 进程数 | `data.workers_per_gpu` | 2 |
| 各阶段 DataLoader 设置 | `data.train_dataloader`、`data.val_dataloader`、`data.test_dataloader` | 验证和测试 batch size 均为 1 |
| 训练日志间隔 | `log_config.interval` | 每 50 次更新，轮末补记 |
| 验证和四联图间隔 | `evaluation.interval` | 每 1 轮 |
| 常规 checkpoint 间隔 | `checkpoint_config.interval` | 每 1 轮，保留最近 3 份 |

每次验证计算 mIoU、mDice 和 mFscore，并按 mIoU 保存最佳权重。默认输出目录是 `work_dirs/segman_t_general_50epochs`。常规 checkpoint 按轮编号为 `epoch_1.pth`、`epoch_2.pth` 等，`latest.pth` 是最新保存的一份。继续训练时用 `--resume-from` 恢复同一实验的模型、优化器和进度。如需超过 50 轮，再调大 `runner.max_epochs`。

```powershell
python tools/train.py local_configs/segman/tiny/segman_t_general.py --seed 42 --resume-from work_dirs/segman_t_general_50epochs/latest.pth
```

### 使用 SegMAN-B 训练 General

[Base 配置](../local_configs/segman/base/segman_b_general.py) 使用 `C:\ZZworkspace\zyworkspace\project\weights\cv\SegMAN\SegMAN_Encoder_b.pth.tar` 初始化编码器。输入为 512 × 512，单张 GPU 每批直接处理 8 张图，每批更新一次参数。训练 50 轮，每 50 次更新记录训练日志，每轮验证一次并保存四张固定四联图。

```powershell
cd C:\ZZworkspace\zyworkspace\project\codes\dis\SegMAN\segmentation
python tools/train.py local_configs/segman/base/segman_b_general.py --gpu-id 0 --seed 42
```

Base 的默认输出目录是 `work_dirs/segman_b_general_50epochs/`，TensorBoard 读取其中的 `tf_logs/`。测试时使用同一份 Base 配置和该目录中的分割 checkpoint。

## 5. TensorBoard 与固定四联图（segmentation 目录）

另开一个 **Anaconda PowerShell Prompt** 启动 TensorBoard，在浏览器打开 `http://localhost:6006`。

```powershell
conda activate segman
cd C:\ZZworkspace\zyworkspace\project\codes\dis\SegMAN\segmentation
tensorboard --logdir work_dirs/segman_t_general_50epochs/tf_logs --host 127.0.0.1 --port 6006
```

在 **Scalars** 查看训练 loss、学习率和验证指标，在 **Images** 查看 `val/preview/` 下的四个样本。每轮验证后生成四张 PNG，每张从左到右显示**原图、GT、预测、预测叠加原图**。第 1 轮的图片位于 `work_dirs/segman_t_general_50epochs/val_vis/epoch_0001/`。

- `F47Mosaic_F47D001008_00000018.png`
- `F48Mosaic_F48D001002_00000043.png`
- `F48Mosaic_F48D002001_00000057.png`
- `F48Mosaic_F48D002003_00000091.png`

默认每轮都在完整验证集的 1,878 张图上计算指标。用 `data.val_dataloader.indices` 指定验证子集时，四张预览图仍从完整验证集读取。

## 6. 测试（segmentation 目录）

用最新权重在 test 拆分的 4,802 张图上评估 mIoU、mDice、mFscore 和分类别指标。指标 JSON 会写入 `work_dirs/segman_t_general_50epochs/test/`。

```powershell
Set-Location "C:\ZZworkspace\zyworkspace\project\codes\dis\SegMAN\segmentation"
python tools/test.py local_configs/segman/tiny/segman_t_general.py --checkpoint work_dirs/segman_t_general_50epochs/latest.pth --eval mIoU mDice mFscore --work-dir work_dirs/segman_t_general_50epochs/test
```

测试最佳权重时，用 `Get-ChildItem work_dirs/segman_t_general_50epochs -Filter "best_mIoU_epoch_*.pth"` 查看文件名，然后把命令中的 `work_dirs/segman_t_general_50epochs/latest.pth` 替换成对应路径。要额外保存测试预测叠加图，在命令末尾加 `--show-dir work_dirs/segman_t_general_50epochs/test/overlays`。

## 7. 小样本跑通流程（segmentation 目录）

下面用 16 张图训练 1 轮，共更新两次参数，随后用两张验证图计算指标并生成四张固定四联图，最后用两张测试图跑测试。

```powershell
Set-Location "C:\ZZworkspace\zyworkspace\project\codes\dis\SegMAN\segmentation"
New-Item -ItemType Directory -Force work_dirs/segman_t_general_smoke | Out-Null
Get-ChildItem "C:\ZZworkspace\zyworkspace\project\datasets\dis\General\train\image" -File -Filter "*.tif" | Sort-Object Name | Select-Object -First 16 -ExpandProperty BaseName | Set-Content -Encoding ascii work_dirs/segman_t_general_smoke/train.txt
Get-ChildItem "C:\ZZworkspace\zyworkspace\project\datasets\dis\General\test\image" -File -Filter "*.tif" | Sort-Object Name | Select-Object -First 2 -ExpandProperty BaseName | Set-Content -Encoding ascii work_dirs/segman_t_general_smoke/test.txt

python tools/train.py local_configs/segman/tiny/segman_t_general.py --work-dir work_dirs/segman_t_general_smoke --seed 42 --cfg-options runner.max_epochs=1 "data.train.split=C:/ZZworkspace/zyworkspace/project/codes/dis/SegMAN/segmentation/work_dirs/segman_t_general_smoke/train.txt" "data.val_dataloader.indices=[4,5]"
python tools/test.py local_configs/segman/tiny/segman_t_general.py --checkpoint work_dirs/segman_t_general_smoke/latest.pth --eval mIoU mDice mFscore --work-dir work_dirs/segman_t_general_smoke/test --cfg-options "data.test.split=C:/ZZworkspace/zyworkspace/project/codes/dis/SegMAN/segmentation/work_dirs/segman_t_general_smoke/test.txt"
```

训练输出在 `work_dirs/segman_t_general_smoke`。四张 PNG 位于 `val_vis/epoch_0001/`，TensorBoard 事件写到 `tf_logs/`，测试指标 JSON 写到 `test/`。
