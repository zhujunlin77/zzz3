import os
import json
import cv2
from datetime import datetime
import numpy as np
import random
import re  # 导入正则表达式模块
from lib.utils.opts import opts

def robust_yolo_to_coco_converter():
    """
    一个健壮的、支持多分辨率和条件抽帧的YOLO到COCO格式转换器。
    - [核心功能] 自动读取每张图片的实际分辨率，不再依赖固定的IMAGE_W/H。
    - [核心功能] 仅对名称不符合 'data' + 数字 格式的文件夹进行抽帧。
    - 能够自动发现数据文件夹。
    - 用户只需明确指定测试集文件夹。
    - 支持多类别。
    """
    # ==================== [修复 argparse 错误] ====================
    # 传入空列表，防止解析不必要的命令行参数
    try:
        opt = opts().parse([])
    except SystemExit:
        # argparse in notebook environments can cause sys.exit().
        # We can create a simple object to hold default paths.
        class SimpleOpts:
            def __init__(self):
                # 在这里设置一个默认的数据目录，以防opts初始化失败
                self.data_dir = '/root/autodl-tmp/mix_scene'
        print("argparse exited, using default paths.")
        opt = SimpleOpts()
    except Exception as e:
        print(f"无法初始化opts，将使用默认路径：{e}")
        class SimpleOpts:
            def __init__(self):
                self.data_dir = '/root/autodl-tmp/mix_scene'
        opt = SimpleOpts()
    # ==========================================================

    # 1. 基础路径配置
    scene_dir = getattr(opt, 'data_dir', '/root/autodl-tmp/mix_scene')
    target_dir = scene_dir
    images_base_dir = os.path.join(scene_dir, "images")
    labels_base_dir = os.path.join(scene_dir, "labels")

    # 2. 数据集划分
    print("🚀 步骤1: 动态发现文件夹并划分数据集...")
    if not os.path.exists(images_base_dir):
        print(f"❌ 错误: 'images' 目录不存在于: {images_base_dir}")
        return False
    
    all_data_folders = sorted([d for d in os.listdir(images_base_dir) if os.path.isdir(os.path.join(images_base_dir, d))])
    print(f"🔍 在 '{images_base_dir}' 中发现 {len(all_data_folders)} 个数据文件夹。")
    
    # --- 用户需要在此处定义测试集 ---
    test_data_folders = [
        'wg2022_ir_020_split_01',
        'wg2022_ir_020_split_03',
        'wg2022_ir_020_split_07',
        # ... 在此继续添加您所有的测试文件夹 ...
    ]
    test_data_folders = all_data_folders
    train_data_folders = [f for f in all_data_folders if f not in test_data_folders]
    train_data_folders = ['data1','data2','data3','data1001','data1002','data1003',
        'wg2022_ir_020_split_01',
        'wg2022_ir_020_split_03',
        'wg2022_ir_020_split_07',
        # ... 在此继续添加您所有的测试文件夹 ...
    ]
    test_data_folders = train_data_folders

    print(f"✅ 数据集划分完成: {len(train_data_folders)} 训练集, {len(test_data_folders)} 测试集")

    annotations_output_dir = os.path.join(target_dir, "annotations")
    os.makedirs(annotations_output_dir, exist_ok=True)

    def create_coco_format():
        return {
            "info": {"description": "Custom Dataset", "version": "1.0", "year": datetime.now().year,
                     "date_created": datetime.now().strftime("%Y-%m-%d")},
            "licenses": [{"id": 1, "name": "Unknown"}], "images": [], "annotations": [], "categories": []
        }

    # 3. 类别定义
    print("\n🔍 步骤2: 定义类别信息...")
    yolo_class_names = ["drone", "car", "ship", "bus", "pedestrian", "cyclist"]
    categories = [{"id": i + 1, "name": name, "supercategory": "object"} for i, name in enumerate(yolo_class_names)]
    yolo_to_coco_mapping = {i: i + 1 for i in range(len(yolo_class_names))}
    print(f"📋 成功定义 {len(categories)} 个类别。")

    def process_dataset(data_folders_list, dataset_type, coco_data, frame_skip_rate_default=3):
        """
        处理指定的数据集文件夹列表。
        - 只有当文件夹名称不符合 'data' + 数字 的格式时，才应用抽帧。
        - 动态读取每张图片的实际分辨率。
        """
        print(f"\n  📂 正在处理 {dataset_type} 数据集 ({len(data_folders_list)} 个文件夹)...")
        print(f"   - 默认抽帧率 (当文件夹名称不匹配时): 1/{frame_skip_rate_default}")
            
        image_id_counter = len(coco_data["images"]) + 1
        annotation_id_counter = len(coco_data["annotations"]) + 1

        for data_folder in data_folders_list:
            image_dir = os.path.join(images_base_dir, data_folder)
            label_dir = os.path.join(labels_base_dir, data_folder)
            
            if not os.path.exists(image_dir) or not os.path.exists(label_dir):
                print(f"   - ⚠️  跳过 '{data_folder}' (目录不存在)")
                continue

            # ==================== [条件抽帧逻辑] ====================
            # 使用正则表达式匹配 'data' 后跟任意数字的模式
            is_no_skip_folder = re.match(r'^data\d+$', data_folder)
            
            if is_no_skip_folder:
                current_frame_skip_rate = 1 # 不抽帧
                # print(f"   -> 文件夹 '{data_folder}' 匹配模式，不抽帧 (1/1)。") # 可选的详细日志
            else:
                current_frame_skip_rate = frame_skip_rate_default
                # if current_frame_skip_rate > 1:
                #     print(f"   -> 文件夹 '{data_folder}' 不匹配，应用抽帧 (1/{current_frame_skip_rate})。") # 可选的详细日志
            # ==========================================================

            def natural_sort_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]
            image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))], key=natural_sort_key)

            for image_index, image_file in enumerate(image_files):
                if image_index % current_frame_skip_rate != 0:
                    continue

                label_full_path = os.path.join(label_dir, os.path.splitext(image_file)[0] + '.txt')
                if not os.path.exists(label_full_path):
                    continue

                # 动态读取图像尺寸
                image_full_path = os.path.join(image_dir, image_file)
                try:
                    img = cv2.imread(image_full_path)
                    if img is None:
                        continue
                    image_h, image_w = img.shape[:2]
                except Exception:
                    continue

                relative_image_path = os.path.join('images', data_folder, image_file).replace(os.sep, '/')
                
                image_info = {
                    "id": image_id_counter, "width": image_w, "height": image_h,
                    "file_name": relative_image_path, "license": 1, "original_file": image_file
                }
                coco_data["images"].append(image_info)

                with open(label_full_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        yolo_class_id, x_c, y_c, w, h = map(float, parts[:5])
                        yolo_class_id = int(yolo_class_id)
                        
                        # 使用动态尺寸进行转换
                        abs_w, abs_h = w * image_w, h * image_h
                        x_min, y_min = (x_c * image_w) - (abs_w / 2), (y_c * image_h) - (abs_h / 2)
                        
                        coco_category_id = yolo_to_coco_mapping.get(yolo_class_id)
                        if coco_category_id is None: continue

                        annotation = {
                            "id": annotation_id_counter, "image_id": image_id_counter,
                            "category_id": coco_category_id, "bbox": [x_min, y_min, abs_w, abs_h],
                            "area": abs_w * abs_h, "iscrowd": 0, "segmentation": []
                        }
                        coco_data["annotations"].append(annotation)
                        annotation_id_counter += 1
                image_id_counter += 1

    # 4. 生成数据集
    # 生成训练集（应用条件抽帧）
    train_coco = create_coco_format()
    train_coco["categories"] = categories
    process_dataset(train_data_folders, "train", train_coco, frame_skip_rate_default=3) 

    # 生成测试集（不抽帧）
    test_coco = create_coco_format()
    test_coco["categories"] = categories
    process_dataset(test_data_folders, "test", test_coco, frame_skip_rate_default=1) 

    # 5. 保存文件
    print("\n🔧 步骤3: 保存注释文件...")
    for (name, data) in [("train", train_coco), ("test", test_coco), ("val", test_coco)]:
        file_path = os.path.join(annotations_output_dir, f"instances_{name}2017.json")
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(
            f"✅ {name.capitalize()}集注释文件: {file_path} ({len(data['images'])} 图像, {len(data['annotations'])} 标注)")

    print("\n🎉 数据转换完成!")
    return True


if __name__ == "__main__":
    if robust_yolo_to_coco_converter():
        print("\n✅ 转换成功！现在可以开始训练。")
    else:
        print("\n❌ 转换失败，请检查错误日志。")