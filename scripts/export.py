"""
YOLOv8 模型导出脚本
支持导出为 ONNX、TorchScript、TensorRT 等格式
"""
from ultralytics import YOLO
import argparse
from pathlib import Path


def export_model(
    model_path: str,
    format: str = "onnx",
    imgsz: int = 640,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True
):
    """
    导出 YOLOv8 模型
    
    Args:
        model_path: 训练好的模型路径 (.pt)
        format: 导出格式 (onnx/torchscript/engine/coreml/ncnn)
        imgsz: 输入图片大小
        half: 是否使用 FP16 量化
        dynamic: 是否使用动态输入尺寸
        simplify: 是否简化 ONNX 模型
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    
    print(f"\n导出配置:")
    print(f"  格式: {format}")
    print(f"  图片大小: {imgsz}")
    print(f"  FP16: {half}")
    print(f"  动态尺寸: {dynamic}")
    print(f"  简化模型: {simplify}")
    
    # 导出模型
    export_path = model.export(
        format=format,
        imgsz=imgsz,
        half=half,
        dynamic=dynamic,
        simplify=simplify
    )
    
    print(f"\n导出完成!")
    print(f"导出模型路径: {export_path}")
    
    return export_path


def export_all_formats(model_path: str, imgsz: int = 640):
    """导出所有常用格式"""
    formats = ["onnx", "torchscript", "engine"]
    
    for fmt in formats:
        print(f"\n{'='*50}")
        print(f"导出格式: {fmt}")
        print('='*50)
        try:
            export_model(model_path, format=fmt, imgsz=imgsz)
        except Exception as e:
            print(f"导出 {fmt} 失败: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 YOLOv8 模型")
    parser.add_argument("--model", required=True,
                       help="训练好的模型路径 (.pt)")
    parser.add_argument("--format", default="onnx",
                       choices=["onnx", "torchscript", "engine", "coreml", "ncnn"],
                       help="导出格式")
    parser.add_argument("--imgsz", type=int, default=640,
                       help="输入图片大小")
    parser.add_argument("--half", action="store_true",
                       help="使用 FP16 量化")
    parser.add_argument("--dynamic", action="store_true",
                       help="使用动态输入尺寸")
    parser.add_argument("--no-simplify", action="store_true",
                       help="不简化 ONNX 模型")
    parser.add_argument("--all", action="store_true",
                       help="导出所有常用格式")
    
    args = parser.parse_args()
    
    # 获取脚本所在目录的父目录作为项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 处理模型路径
    model_path = args.model
    if not Path(model_path).is_absolute():
        model_path = str(project_root / model_path)
    
    if args.all:
        export_all_formats(model_path, args.imgsz)
    else:
        export_model(
            model_path=model_path,
            format=args.format,
            imgsz=args.imgsz,
            half=args.half,
            dynamic=args.dynamic,
            simplify=not args.no_simplify
        )
