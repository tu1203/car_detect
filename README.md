# Car Detect - 车辆检测项目

基于 YOLOv8 的汽车目标检测项目，支持训练、导出和实时推理。支持 NVIDIA Jetson 设备部署，新增 TensorRT 推理加速。

## 项目结构

```
car_detect/
├── data/                    # 数据集
│   ├── images/              # 图片文件
│   │   ├── train/           # 训练集
│   │   ├── val/             # 验证集
│   │   └── test/            # 测试集
│   ├── labels/              # YOLO格式标注
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml            # 数据集配置
├── scripts/                 # 功能脚本
│   ├── train.py             # 训练
│   ├── export.py            # 模型导出
│   ├── detect.py            # 推理检测
│   ├── gui.py               # GUI界面
│   └── download_dataset.py  # 数据集下载
├── models/                  # 模型输出目录
│   ├── weights/             # 权重文件 (.pt, .onnx, .engine)
│   └── logs/                # 训练日志 (图片、CSV、配置)
├── requirements.txt         # Python依赖
└── .gitignore
```

## 环境配置

### 标准环境 (x86_64 / 桌面GPU)

#### 1. 创建虚拟环境

```bash
cd car_detect
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：
- `ultralytics` - YOLOv8 框架
- `opencv-python` - 图像处理
- `torch` / `torchvision` - 深度学习框架

### Jetson 设备环境 (JetPack 6.x)

#### 1. 创建虚拟环境

```bash
cd car_detect
python3 -m venv .venv
source .venv/bin/activate
```

#### 2. 安装系统依赖

```bash
sudo apt-get -y install libopenblas-dev libopenmpi-dev libomp-dev
```

#### 3. 安装 Jetson 专用 PyTorch

从 NVIDIA 官方论坛下载预编译的 wheel 文件（NVIDIA Jetson 专用版本）：

```bash
# 安装 PyTorch (JetPack 6.1)
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# 安装 torchvision (从源码编译，匹配 NVIDIA torch)
pip install torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl
```

#### 4. 安装其他依赖

```bash
pip install -r requirements.txt
```

#### 5. 链接 CUDA 库

NVIDIA torch 依赖 `libcusparseLt.so.0`，需要链接到系统标准路径：

```bash
sudo ln -s /home/nano/car_detect/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0 /usr/lib/aarch64-linux-gnu/
sudo ldconfig
```

#### 6. 链接 TensorRT

Jetson 设备的 TensorRT 通过系统包安装，需要链接到虚拟环境：

```bash
ln -s /usr/lib/python3.10/dist-packages/tensorrt .venv/lib/python3.10/site-packages/
ln -s /usr/lib/python3.10/dist-packages/tensorrt_*.dist-info .venv/lib/python3.10/site-packages/ 2>/dev/null
```

#### Jetson 设备兼容性

| JetPack | PyTorch | torchvision | 来源 |
|---------|---------|-------------|------|
| 6.0 (L4T R36.2/R36.3) | 2.3.0 | 0.18.0 | NVIDIA 官方 wheel |
| 6.1 (L4T R36.4) | 2.5.0a0 (nv24.08) | 0.20.0 (源码编译) | 项目提供 |

> **注意**：NVIDIA 官方源只有 torch，没有 torchvision。torchvision 必须从源码编译才能与 NVIDIA torch 兼容。

### Python 版本要求

- 最低：Python 3.8
- 推荐：Python 3.10+

## 使用教程

### 1. 准备数据集

#### 使用现有数据集

项目包含 200 张汽车检测图片（160张训练 + 40张验证），从 Roboflow 数据集筛选 car 类别。

#### 自定义数据集

数据集必须遵循 YOLO 格式：

```
data/
├── images/train/    # 训练图片 (.jpg)
├── labels/train/    # 训练标注 (.txt)
├── images/val/      # 验证图片
└── labels/val/      # 验证标注
```

每个标注文件格式（一行一个目标）：

```
class_id  x_center  y_center  width  height
```

所有坐标都是**归一化**的（0~1之间）。

示例标注文件 `000000000001.txt`：
```
0 0.5 0.5 0.3 0.4
0 0.2 0.3 0.1 0.2
```

第一行表示：类别0（car），中心点在图片50%宽度、50%高度处，框宽30%、高40%。

#### 修改 data.yaml

```yaml
path: /home/tu/opencode_ws/car_detect/data
train: images/train
val: images/val
test: images/test

names:
  0: car
```

### 2. 训练模型

```bash
# 基础训练
python scripts/train.py

# 自定义参数训练
python scripts/train.py \
    --model-size s \      # 模型大小: n/s/m/l/x
    --epochs 100 \        # 训练轮数
    --batch-size 16 \     # 批次大小
    --imgsz 640 \         # 输入图片尺寸
    --device 0            # GPU编号，cpu则用cpu
```

训练完成后，模型保存在 `models/weights/` 目录：
- `best.pt` - 最佳模型
- `last.pt` - 最后一轮模型

#### 模型大小选择

| 模型 | 参数量 | 速度 | 精度 |
|------|--------|------|------|
| yolov8n | 3.2M | 最快 | 最低 |
| yolov8s | 11.2M | 快 | 中等 |
| yolov8m | 25.9M | 中等 | 较高 |
| yolov8l | 43.7M | 慢 | 高 |
| yolov8x | 68.2M | 最慢 | 最高 |

### 3. 验证模型

```bash
python scripts/train.py --validate models/weights/best.pt
```

输出指标：
- `mAP50` - IoU=0.5时的平均精度
- `mAP50-95` - IoU从0.5到0.95的平均精度

### 4. 导出模型

```bash
# 导出为ONNX格式（推荐，通用性强）
python scripts/export.py --model models/weights/best.pt --format onnx

# 导出为TensorRT格式（NVIDIA GPU加速）
python scripts/export.py --model models/weights/best.pt --format engine

# 导出所有常用格式
python scripts/export.py --model models/weights/best.pt --all
```

支持的导出格式：
- `onnx` - 通用格式，跨平台部署
- `torchscript` - PyTorch原生格式
- `engine` - TensorRT格式，NVIDIA GPU加速（推荐用于部署）

### 5. 推理检测

#### 检测单张图片

```bash
python scripts/detect.py \
    --model models/weights/best.pt \
    --source test.jpg \
    --show
```

#### 检测视频

```bash
python scripts/detect.py \
    --model models/weights/best.pt \
    --source video.mp4 \
    --save
```

#### 摄像头实时检测

```bash
python scripts/detect.py \
    --model models/weights/best.pt \
    --realtime
```

按 `q` 退出实时检测。

#### 使用ONNX模型推理

```bash
python scripts/detect.py \
    --model models/weights/best.onnx \
    --source test.jpg
```

#### 使用TensorRT模型推理（推荐用于部署）

```bash
python scripts/detect.py \
    --model models/weights/best.engine \
    --source test.jpg
```

### 6. GUI 界面

启动图形界面进行交互式检测：

```bash
python scripts/gui.py
```

功能：
- 打开图片 - 选择本地图片
- 开始检测 - 运行 YOLOv8 检测
- 保存结果 - 保存标注后的图片
- 置信度滑块 - 调整检测阈值

---

## 训练结果

### 当前模型性能

| 指标 | 值 |
|------|-----|
| mAP50 | 0.897 (89.7%) |
| mAP50-95 | 0.648 (64.8%) |
| Precision | 0.889 |
| Recall | 0.835 |
| 训练轮数 | 50 epochs |
| 训练数据 | 169 张图片 |
| 验证数据 | 40 张图片 |
| 训练设备 | RTX 3050 (4GB) |
| 训练时间 | 约 2 分钟 |

### 模型文件

```
models/weights/
├── best.pt           (6.3MB)  PyTorch格式 - 最佳模型
├── last.pt           (6.3MB)  最后一轮模型
├── best.onnx         (11.7MB) ONNX格式 - 通用部署
└── best.engine       (13.2MB) TensorRT格式 - NVIDIA GPU加速
```

### 推理速度对比

| 格式 | 设备 | 推理速度 | 说明 |
|------|------|----------|------|
| .pt (PyTorch) | Jetson Orin | ~53ms | 基准速度 |
| .engine (TensorRT) | Jetson Orin | ~47ms | 加速 1.46x |
| .pt (PyTorch) | RTX 3050 | ~15ms | 桌面GPU |
| .engine (TensorRT) | RTX 3050 | ~8ms | 桌面GPU加速 |

> **注意**：TensorRT 首次加载需要额外时间，后续推理会更快。小模型（yolov8n）加速效果不如大模型明显。

---

## 常见问题

### 训练时显存不足

```bash
# 减小批次大小
python scripts/train.py --batch-size 8

# 使用更小的模型
python scripts/train.py --model-size n
```

### 检测精度低

1. 增加训练数据量
2. 增加训练轮数：`--epochs 200`
3. 使用更大的模型：`--model-size s` 或 `--model-size m`
4. 检查数据集标注质量

### 推理速度慢

1. 导出 ONNX 或 TensorRT 格式
2. 使用更小的模型
3. 减小输入图片尺寸：`--imgsz 320`

### Jetson 设备相关问题

#### ImportError: libcusparseLt.so.0

```bash
# 链接到系统标准路径
sudo ln -s /home/nano/car_detect/.venv/lib/python3.10/site-packages/nvidia/cusparselt/lib/libcusparseLt.so.0 /usr/lib/aarch64-linux-gnu/
sudo ldconfig
```

#### RuntimeError: operator torchvision::nms does not exist

torchvision 版本与 NVIDIA torch 不兼容。确保使用项目提供的 torchvision wheel 文件。

#### TensorRT 导出失败

确保已链接 TensorRT 到虚拟环境：
```bash
ln -s /usr/lib/python3.10/dist-packages/tensorrt .venv/lib/python3.10/site-packages/
```

---

## 参考资料

- [YOLOv8 官方文档](https://docs.ultralytics.com/)
- [NVIDIA Jetson PyTorch](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)
- [TensorRT 文档](https://developer.nvidia.com/tensorrt)
