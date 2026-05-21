"""
YOLOv8 车辆检测 GUI 界面
使用 tkinter 实现
"""
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from pathlib import Path
import threading


class CarDetectApp:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("车辆检测系统")
        self.root.geometry("900x700")
        
        self.model = YOLO(model_path)
        self.current_image = None
        self.photo_image = None
        
        self._create_widgets()
    
    def _create_widgets(self):
        # 顶部控制栏
        control_frame = tk.Frame(self.root, pady=10)
        control_frame.pack(fill=tk.X)
        
        # 模型路径显示
        tk.Label(control_frame, text="模型:").pack(side=tk.LEFT, padx=5)
        self.model_label = tk.Label(control_frame, text="best.pt", relief=tk.SUNKEN, width=30)
        self.model_label.pack(side=tk.LEFT, padx=5)
        
        # 置信度滑块
        tk.Label(control_frame, text="置信度:").pack(side=tk.LEFT, padx=5)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.conf_slider = tk.Scale(
            control_frame, from_=0.1, to=1.0, resolution=0.05,
            orient=tk.HORIZONTAL, variable=self.conf_var, length=150
        )
        self.conf_slider.pack(side=tk.LEFT, padx=5)
        
        # 按钮
        btn_frame = tk.Frame(self.root, pady=5)
        btn_frame.pack(fill=tk.X)
        
        tk.Button(btn_frame, text="打开图片", command=self.open_image, width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="开始检测", command=self.run_detection, width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="保存结果", command=self.save_result, width=12).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="清空", command=self.clear, width=12).pack(side=tk.LEFT, padx=10)
        
        # 图片显示区域
        self.canvas = tk.Canvas(self.root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 底部状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)
        
        # 检测结果信息
        self.result_var = tk.StringVar(value="")
        result_label = tk.Label(self.root, textvariable=self.result_var, relief=tk.SUNKEN, anchor=tk.W, height=2)
        result_label.pack(fill=tk.X, side=tk.BOTTOM, padx=10, pady=5)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("所有文件", "*.*")
            ]
        )
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.status_var.set(f"已加载: {file_path}")
            self.result_var.set("")
    
    def display_image(self, image_path):
        img = Image.open(image_path)
        
        # 调整图片大小以适应窗口
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            canvas_width, canvas_height = 860, 500
        
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo_image, anchor=tk.CENTER
        )
    
    def run_detection(self):
        if not hasattr(self, 'current_image_path'):
            messagebox.showwarning("提示", "请先打开图片")
            return
        
        self.status_var.set("正在检测...")
        self.root.update()
        
        conf = self.conf_var.get()
        
        results = self.model.predict(
            source=self.current_image_path,
            conf=conf,
            save=False,
            verbose=False
        )
        
        # 绘制检测结果
        result = results[0]
        annotated_frame = result.plot()
        
        # 保存结果图片
        self.result_image = annotated_frame
        
        # 转换为PIL图片显示
        annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(annotated_rgb)
        
        # 调整大小
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        img_ratio = img.width / img.height
        canvas_ratio = canvas_width / canvas_height
        
        if img_ratio > canvas_ratio:
            new_width = canvas_width
            new_height = int(canvas_width / img_ratio)
        else:
            new_height = canvas_height
            new_width = int(canvas_height * img_ratio)
        
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        self.photo_image = ImageTk.PhotoImage(img)
        
        self.canvas.delete("all")
        self.canvas.create_image(
            canvas_width // 2, canvas_height // 2,
            image=self.photo_image, anchor=tk.CENTER
        )
        
        # 显示检测结果
        boxes = result.boxes
        result_text = f"检测到 {len(boxes)} 个目标"
        if len(boxes) > 0:
            details = []
            for box in boxes:
                cls_name = self.model.names[int(box.cls[0])]
                conf_score = float(box.conf[0])
                details.append(f"{cls_name}({conf_score:.2f})")
            result_text += ": " + ", ".join(details)
        
        self.result_var.set(result_text)
        self.status_var.set("检测完成")
    
    def save_result(self):
        if not hasattr(self, 'result_image'):
            messagebox.showwarning("提示", "没有检测结果可保存")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="保存结果",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")]
        )
        if file_path:
            cv2.imwrite(file_path, self.result_image)
            self.status_var.set(f"结果已保存: {file_path}")
    
    def clear(self):
        self.canvas.delete("all")
        self.result_var.set("")
        self.status_var.set("就绪")
        if hasattr(self, 'current_image_path'):
            del self.current_image_path
        if hasattr(self, 'result_image'):
            del self.result_image


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="车辆检测 GUI")
    parser.add_argument("--model", default="models/weights/best.pt",
                       help="模型路径")
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_path = str(project_root / args.model)
    
    root = tk.Tk()
    app = CarDetectApp(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
