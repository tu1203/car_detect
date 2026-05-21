# Car Detect - 车辆检测项目

基于 YOLOv8 的汽车目标检测项目，支持训练、导出和实时推理。

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
│   └── download_dataset.py  # 数据集下载
├── models/                  # 模型输出目录
├── requirements.txt         # Python依赖
└── .gitignore
```

## 环境配置

### 1. 创建虚拟环境

```bash
cd car_detect
python3 -m venv .venv
source .venv/bin/activate
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖项：
- `ultralytics` - YOLOv8 框架
- `opencv-python` - 图像处理
- `torch` / `torchvision` - 深度学习框架

### Python 版本要求

- 最低：Python 3.8
- 推荐：Python 3.10+

## 使用教程

### 1. 准备数据集

#### 使用现有数据集

项目已包含 COCO128 筛选的汽车数据（12张图片）。

#### 下载更多数据

```bash
python scripts/download_dataset.py --max-images 1000
```

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

训练完成后，模型保存在 `models/car_detect/weights/` 目录：
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
python scripts/train.py --validate models/car_detect/weights/best.pt
```

输出指标：
- `mAP50` - IoU=0.5时的平均精度
- `mAP50-95` - IoU从0.5到0.95的平均精度

### 4. 导出模型

```bash
# 导出为ONNX格式（推荐，通用性强）
python scripts/export.py --model models/car_detect/weights/best.pt --format onnx

# 导出所有常用格式
python scripts/export.py --model models/car_detect/weights/best.pt --all
```

支持的导出格式：
- `onnx` - 通用格式，跨平台部署
- `torchscript` - PyTorch原生格式
- `engine` - TensorRT格式，NVIDIA GPU加速

### 5. 推理检测

#### 检测单张图片

```bash
python scripts/detect.py \
    --model models/car_detect/weights/best.pt \
    --source test.jpg \
    --show
```

#### 检测视频

```bash
python scripts/detect.py \
    --model models/car_detect/weights/best.pt \
    --source video.mp4 \
    --save
```

#### 摄像头实时检测

```bash
python scripts/detect.py \
    --model models/car_detect/weights/best.pt \
    --realtime
```

按 `q` 退出实时检测。

#### 使用ONNX模型推理

```bash
python scripts/detect.py \
    --model models/car_detect/weights/best.onnx \
    --source test.jpg
```

---

## YOLOv8 原理

### 什么是 YOLO

YOLO（You Only Look Once）是一种单阶段目标检测算法，与传统方法不同，它只需一次前向传播即可同时预测多个目标的位置和类别。

### 核心思想

传统目标检测方法（如 R-CNN 系列）：
1. 先生成候选区域（Region Proposals）
2. 再对每个区域分类

YOLO 的方法：
1. 将图片划分为 S×S 网格
2. 每个网格直接预测边界框和类别
3. 一次推理完成所有检测

### YOLOv8 网络结构

```
输入图片 (640×640×3)
    │
    ▼
┌─────────┐
│  Backbone │  特征提取（CSPDarknet）
│  主干网络  │  提取多尺度特征
└────┬────┘
     │
     ▼
┌─────────┐
│   Neck   │  特征融合（PAN-FPN）
│  特征金字塔 │  融合不同尺度信息
└────┬────┘
     │
     ▼
┌─────────┐
│   Head   │  检测头（Decoupled Head）
│  检测头   │  预测框+类别
└────┬────┘
     │
     ▼
输出：边界框坐标 + 置信度 + 类别概率
```

### 关键组件

#### 1. Backbone（主干网络）

YOLOv8 使用改进的 CSPDarknet：
- **CBS 模块**：Conv + BatchNorm + SiLU 激活
- **C2f 模块**：跨阶段部分连接，减少计算量同时保持特征丰富度
- **SPPF**：空间金字塔池化，增强多尺度感受野

#### 2. Neck（特征融合）

使用 PAN-FPN（路径聚合特征金字塔）：
- 自顶向下传递高层语义信息
- 自底向上传递低层位置信息
- 实现多尺度特征融合

#### 3. Head（检测头）

YOLOv8 采用**解耦头**设计：
- 分类分支：预测类别概率
- 回归分支：预测边界框坐标

与 YOLOv5 不同，YOLOv8 **取消了锚框（Anchor-Free）**，直接预测目标中心点到网格边界的距离。

### 训练过程

#### 损失函数

YOLOv8 使用三种损失的加权和：

```
总损失 = λ₁ × 分类损失 + λ₂ × 定位损失 + λ₃ × DFL损失
```

1. **分类损失**（BCE Loss）：二元交叉熵，判断每个类别的有无
2. **定位损失**（IoU Loss）：计算预测框与真实框的重叠度
3. **DFL 损失**：分布式焦点损失，更精确地定位边界

#### 数据增强

YOLOv8 内置多种数据增强：
- **Mosaic**：将4张图片拼接，增加背景多样性
- **MixUp**：混合两张图片，平滑类别边界
- **随机仿射变换**：旋转、缩放、翻转
- **颜色扰动**：调整亮度、对比度、饱和度

### 推理过程

1. **输入预处理**
   - 图片缩放至 640×640
   - 归一化到 [0, 1]

2. **特征提取与融合**
   - Backbone 提取多尺度特征
   - Neck 融合不同层级特征

3. **检测头预测**
   - 每个网格预测边界框和类别概率
   - 输出格式：`[batch, num_boxes, 5+num_classes]`

4. **后处理**
   - **置信度过滤**：丢弃低置信度预测
   - **NMS（非极大值抑制）**：去除重叠框，保留最佳预测

### NMS 算法

```
1. 按置信度排序所有预测框
2. 选择置信度最高的框
3. 计算该框与其他框的 IoU
4. 移除 IoU > 阈值的重叠框
5. 重复步骤2-4，直到没有框剩余
```

### YOLOv8 改进点（相比 YOLOv5）

| 特性 | YOLOv5 | YOLOv8 |
|------|--------|--------|
| 锚框 | Anchor-Based | Anchor-Free |
| 检测头 | 耦合头 | 解耦头 |
| 主干模块 | C3 | C2f |
| 损失函数 | CIoU | DFL + IoU |
| 特征融合 | PAN | PAN + SPPF |

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
