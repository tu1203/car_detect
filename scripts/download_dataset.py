"""
从 OpenImages 下载汽车类别数据集
使用 OIDv4_ToolKit 或直接下载标注
"""
import os
import subprocess
import sys
from pathlib import Path


def install_oid_toolkit():
    """安装 OIDv4_ToolKit"""
    print("正在安装 OIDv4_ToolKit...")
    subprocess.run([sys.executable, "-m", "pip", "install", "oidv4"], check=True)


def download_car_dataset(output_dir, max_images=1000):
    """
    从 OpenImages 下载汽车数据集
    
    Args:
        output_dir: 输出目录
        max_images: 最大下载图片数
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用 oidv4 命令行工具下载
    # Car 在 OpenImages 中的 class name 是 "Car"
    cmd = [
        "oidv4",
        "download",
        "--dataset", str(output_dir),
        "--classes", "Car",
        "--type_csv", "train",
        "--limit", str(max_images)
    ]
    
    print(f"开始下载汽车数据集，最多 {max_images} 张图片...")
    print(f"下载目录: {output_dir}")
    
    try:
        subprocess.run(cmd, check=True)
        print("下载完成!")
    except subprocess.CalledProcessError as e:
        print(f"下载失败: {e}")
        print("\n备选方案: 手动下载")
        print("1. 访问 https://storage.googleapis.com/openimages/web/index.html")
        print("2. 搜索 'Car' 类别")
        print("3. 下载图片和标注文件")


def convert_to_yolo_format(oid_dir, output_dir):
    """
    将 OpenImages 标注转换为 YOLO 格式
    
    Args:
        oid_dir: OIDv4_ToolKit 下载的目录
        output_dir: YOLO 格式输出目录
    """
    output_dir = Path(output_dir)
    images_dir = output_dir / "images" / "train"
    labels_dir = output_dir / "labels" / "train"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # 查找所有标注文件
    oid_path = Path(oid_dir)
    label_files = list(oid_path.rglob("*.txt"))
    
    print(f"找到 {len(label_files)} 个标注文件")
    
    for label_file in label_files:
        # 读取 OID 标注 (class_id x_min y_min x_max y_max)
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        # 转换为 YOLO 格式 (class_id x_center y_center width height)
        yolo_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = 0  # Car 类别 ID 为 0
                x_min, y_min, x_max, y_max = map(float, parts[1:5])
                
                # 计算中心点和宽高
                x_center = (x_min + x_max) / 2
                y_center = (y_min + y_max) / 2
                width = x_max - x_min
                height = y_max - y_min
                
                yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}")
        
        # 保存 YOLO 标注
        label_name = label_file.stem + ".txt"
        output_label = labels_dir / label_name
        
        with open(output_label, 'w') as f:
            f.write('\n'.join(yolo_lines))
        
        # 复制对应的图片
        image_file = label_file.with_suffix('.jpg')
        if image_file.exists():
            import shutil
            shutil.copy2(image_file, images_dir / image_file.name)
    
    print(f"转换完成!")
    print(f"图片目录: {images_dir}")
    print(f"标注目录: {labels_dir}")


def create_dataset_yaml(output_dir):
    """创建 data.yaml 配置文件"""
    output_dir = Path(output_dir)
    yaml_content = f"""# 车辆识别数据集配置
path: {output_dir.absolute()}
train: images/train
val: images/val
test: images/test

# 类别名称
names:
  0: car
"""
    
    yaml_file = output_dir / "data.yaml"
    with open(yaml_file, 'w') as f:
        f.write(yaml_content)
    
    print(f"配置文件已创建: {yaml_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载汽车数据集")
    parser.add_argument("--output", "-o", default="../data",
                       help="输出目录 (默认: ../data)")
    parser.add_argument("--max-images", "-n", type=int, default=1000,
                       help="最大下载图片数 (默认: 1000)")
    parser.add_argument("--install-toolkit", action="store_true",
                       help="仅安装 OIDv4_ToolKit")
    
    args = parser.parse_args()
    
    if args.install_toolkit:
        install_oid_toolkit()
    else:
        # 获取脚本所在目录的父目录作为项目根目录
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_dir = project_root / args.output
        
        download_car_dataset(output_dir, args.max_images)
        create_dataset_yaml(output_dir)
