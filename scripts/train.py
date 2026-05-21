"""
YOLOv8 车辆检测训练脚本
"""
from ultralytics import YOLO
import argparse
from pathlib import Path


def train(
    data_yaml: str = "../data/data.yaml",
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch_size: int = 16,
    device: str = "0",
    project: str = "../models",
    name: str = "car_detect"
):
    """
    训练 YOLOv8 模型
    
    Args:
        data_yaml: 数据集配置文件路径
        model_size: 模型大小 (n/s/m/l/x)
        epochs: 训练轮数
        imgsz: 输入图片大小
        batch_size: 批次大小
        device: 设备 (0=GPU, cpu=CPU)
        project: 模型保存项目目录
        name: 实验名称
    """
    # 加载预训练模型
    model_name = f"yolov8{model_size}.pt"
    print(f"加载预训练模型: {model_name}")
    model = YOLO(model_name)
    
    # 开始训练
    print(f"\n开始训练:")
    print(f"  数据集: {data_yaml}")
    print(f"  模型: {model_name}")
    print(f"  轮数: {epochs}")
    print(f"  图片大小: {imgsz}")
    print(f"  批次大小: {batch_size}")
    print(f"  设备: {device}")
    print()
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        verbose=True,
        seed=42,
        deterministic=True,
    )
    
    print(f"\n训练完成!")
    print(f"模型保存位置: {project}/{name}")
    
    return results


def validate(model_path: str, data_yaml: str):
    """验证模型性能"""
    print(f"\n验证模型: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=data_yaml)
    
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 YOLOv8 车辆检测模型")
    parser.add_argument("--data", default="../data/data.yaml",
                       help="数据集配置文件路径")
    parser.add_argument("--model-size", default="n", choices=["n", "s", "m", "l", "x"],
                       help="模型大小 (n=最快, x=最精确)")
    parser.add_argument("--epochs", type=int, default=100,
                       help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="输入图片大小")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="批次大小")
    parser.add_argument("--device", default="0",
                       help="设备 (0=GPU, cpu=CPU)")
    parser.add_argument("--project", default="../models",
                       help="模型保存目录")
    parser.add_argument("--name", default="car_detect",
                       help="实验名称")
    parser.add_argument("--validate", type=str, default=None,
                       help="验证模型路径")
    
    args = parser.parse_args()
    
    # 获取脚本所在目录的父目录作为项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    data_yaml = str(project_root / args.data.lstrip("../"))
    project = str(project_root / args.project.lstrip("../"))
    
    if args.validate:
        validate(args.validate, data_yaml)
    else:
        train(
            data_yaml=data_yaml,
            model_size=args.model_size,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
            device=args.device,
            project=project,
            name=args.name
        )
