"""
YOLOv8 车辆检测推理脚本
支持图片、视频、摄像头实时检测
"""
from ultralytics import YOLO
import cv2
import argparse
from pathlib import Path


def detect_image(
    model: YOLO,
    image_path: str,
    conf: float = 0.25,
    save: bool = True,
    show: bool = False
):
    """
    检测单张图片
    
    Args:
        model: YOLO 模型
        image_path: 图片路径
        conf: 置信度阈值
        save: 是否保存结果
        show: 是否显示结果
    """
    print(f"检测图片: {image_path}")
    
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save,
        show=show
    )
    
    # 打印检测结果
    for result in results:
        boxes = result.boxes
        print(f"\n检测到 {len(boxes)} 个目标:")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf_score = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            print(f"  类别: {model.names[cls_id]}, 置信度: {conf_score:.2f}")
            print(f"  位置: [{xyxy[0]:.0f}, {xyxy[1]:.0f}, {xyxy[2]:.0f}, {xyxy[3]:.0f}]")
    
    return results


def detect_video(
    model: YOLO,
    video_path: str,
    conf: float = 0.25,
    save: bool = True,
    show: bool = True
):
    """
    检测视频
    
    Args:
        model: YOLO 模型
        video_path: 视频路径 (0 表示摄像头)
        conf: 置信度阈值
        save: 是否保存结果
        show: 是否显示结果
    """
    print(f"检测视频: {video_path}")
    
    results = model.predict(
        source=video_path,
        conf=conf,
        save=save,
        show=show,
        stream=True  # 流式处理，节省内存
    )
    
    for result in results:
        if show:
            # 显示结果
            annotated_frame = result.plot()
            cv2.imshow("Car Detection", annotated_frame)
            
            # 按 'q' 退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if show:
        cv2.destroyAllWindows()
    
    return results


def detect_realtime(
    model: YOLO,
    camera_id: int = 0,
    conf: float = 0.25
):
    """
    摄像头实时检测
    
    Args:
        model: YOLO 模型
        camera_id: 摄像头 ID
        conf: 置信度阈值
    """
    print(f"启动摄像头实时检测 (摄像头 ID: {camera_id})")
    print("按 'q' 退出")
    
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("错误: 无法打开摄像头")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("错误: 无法读取帧")
            break
        
        # 检测
        results = model.predict(
            source=frame,
            conf=conf,
            verbose=False
        )
        
        # 绘制结果
        annotated_frame = results[0].plot()
        
        # 显示
        cv2.imshow("Car Detection", annotated_frame)
        
        # 按 'q' 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def load_model(model_path: str):
    """
    加载模型
    
    Args:
        model_path: 模型路径 (.pt 或 .onnx)
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    print(f"模型加载成功!")
    print(f"类别名称: {model.names}")
    
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 车辆检测推理")
    parser.add_argument("--model", required=True,
                       help="模型路径 (.pt 或 .onnx)")
    parser.add_argument("--source", default=None,
                       help="输入源 (图片/视频路径，0=摄像头)")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="置信度阈值")
    parser.add_argument("--save", action="store_true", default=True,
                       help="保存检测结果")
    parser.add_argument("--no-save", action="store_true",
                       help="不保存检测结果")
    parser.add_argument("--show", action="store_true",
                       help="显示检测结果")
    parser.add_argument("--realtime", action="store_true",
                       help="摄像头实时检测")
    parser.add_argument("--camera", type=int, default=0,
                       help="摄像头 ID")
    
    args = parser.parse_args()
    
    # 获取脚本所在目录的父目录作为项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    # 处理模型路径
    model_path = args.model
    if not Path(model_path).is_absolute():
        model_path = str(project_root / model_path)
    
    # 加载模型
    model = load_model(model_path)
    
    # 根据参数执行不同检测模式
    if args.realtime:
        detect_realtime(model, args.camera, args.conf)
    elif args.source is not None:
        source = args.source
        if source.isdigit():
            source = int(source)
        
        # 判断是图片还是视频
        if isinstance(source, int) or source.endswith(('.mp4', '.avi', '.mov', '.mkv')):
            detect_video(model, source, args.conf, args.save and not args.no_save, args.show)
        else:
            detect_image(model, source, args.conf, args.save and not args.no_save, args.show)
    else:
        print("错误: 请指定输入源 (--source) 或使用 --realtime 模式")
        print("示例:")
        print("  python detect.py --model ../models/weights/best.pt --source image.jpg")
        print("  python detect.py --model ../models/weights/best.pt --source video.mp4")
        print("  python detect.py --model ../models/weights/best.pt --realtime")
