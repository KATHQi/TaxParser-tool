"""
Tax Document Parser - End-to-End Processing Pipeline
将 Jupyter Notebook 封装成可执行的 Python 脚本
功能: 输入一个文件夹路径,里面可能包含一到多张图像,进行完整的文档解析

处理流程:
1. Step0: 图像归一化大小 (横向1100x1560, 纵向2200x1480)
2. Layout分析: 分出合法布局区域
3. 去水印处理
4. OCR检测识别
5. 结果分组: 根据layout进行分组
6. 表格提取: table/table2区域单独进行表格识别
7. 输出结果: grouped数据+表格dataframe+HTML

Author: Auto-generated from Jupyter Notebook
"""

import os
import sys
import json
import argparse
import copy
import random
import re
import numpy as np
import pandas as pd
import cv2
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
from paddleocr import PPStructure,draw_structure_result,save_structure_res
ppocr_root = '/data/tensorflow/kath/OCR/PaddleOCR-release-2.7'
if ppocr_root not in sys.path:
    sys.path.insert(0, ppocr_root)

# Now we can import from ppstructure submodules
from ppstructure.table.matcher import TableMatch, distance, compute_iou
from tools.infer.predict_system import *
from ppstructure.table.predict_structure import *


# 添加依赖路径
jupyter_funcs_path = os.path.join(os.path.dirname(__file__), 'jupyter', 'funcs')
jupyter_tools_path = os.path.join(os.path.dirname(__file__), 'jupyter', 'tools')
paddle_deploy_path = '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/deploy/python'

if jupyter_funcs_path not in sys.path:
    sys.path.insert(0, jupyter_funcs_path)
if jupyter_tools_path not in sys.path:
    sys.path.insert(0, jupyter_tools_path)
if paddle_deploy_path not in sys.path:
    sys.path.insert(0, paddle_deploy_path)

try:
    from funcs.imagetools import img_show, structure_box_trans, visualize_table_annotation
    from funcs.crop_tables import crop_image_from_boxrow
except ImportError as e:
    print(f"Warning (imagetools/crop_tables): {e}")
    # 定义备用函数
    def img_show(img): pass
    def structure_box_trans(structure, *args): return structure
    def visualize_table_annotation(*args, **kwargs): pass
    def crop_image_from_boxrow(image, row, det_results=None):
        """备用的表格裁剪函数"""
        if det_results is not None:
            return None, None  # 返回 (cropped_img, mapped_results)
        return None  # 只返回裁剪后的图像
    exit()

# 定义本地的 redink_remover 函数，避免导入问题
def redink_remover(img):
    """
    红墨水去除 - 仅对 y 轴 1/2 以下区域只保留红色通道，上半部分保持原图
    Args:
        img: OpenCV 图像 (BGR 格式)
    Returns:
        处理后的图像
    """
    try:
        image = copy.deepcopy(img)
        h = image.shape[0]
        mid = h // 2

        # 下半部分去红处理
        bottom = image[mid:, :, :]
        blue_c, green_c, red_c = cv2.split(bottom)
        processed = np.expand_dims(red_c, axis=2)
        processed = np.concatenate((processed, processed, processed), axis=-1)
        image[mid:, :, :] = processed

        return image
    except Exception as e:
        print(f"警告: 红墨水去除失败 {e}，使用原图")
        return img

# PaddleOCR 依赖
try:
    from paddleocr import PaddleOCR, draw_ocr, PPStructure, draw_structure_result, save_structure_res
except ImportError as e:
    print(f"Warning: PaddleOCR not installed - {e}")

warnings.filterwarnings('ignore')


# ==================== 辅助函数-字符串相似度 ====================

from difflib import SequenceMatcher

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()




def is_page_string(s):
    PAGE_PATTERN = re.compile(
    r"""^
    (                             # 整体开始
        第?                       # 可选的“第”
        (?:\d+|[一二三四五六七八九十百千]+)   # X 页（数字或中文数字）
        页                         # “页”
        (?:                        # 可选：后面接“共 Y 页”
            (?:,|，)?              # 可以有中文逗号
            (?:总共|共)
            (?:\d+|[一二三四五六七八九十百千]+)
            页
        )?
    )
    |
    (?:总共|共)                    # 单独的“总共 Y 页”
    (?:\d+|[一二三四五六七八九十百千]+)
    页
    $""",
    re.X
)
    return bool(PAGE_PATTERN.match(s))

# ==================== 辅助函数 ====================
def extract_page_info(s):
    pattern = re.compile(
        r'第(\d+|[一二三四五六七八九十百千]+)页[^0-9一二三四五六七八九十百千]+(?:总共|共)(\d+|[一二三四五六七八九十百千]+)页'
    )
    m = pattern.search(s)
    if not m:
        return None
    
    x, y = m.groups()
    
    return [x, y]


def reorganize_grouped_data_by_bounds(grouped_data: Dict,
                                     table_upper_bound: float,
                                     table_lower_bound: float,
                                     region_name: str = 'table') -> Tuple[Dict, Dict]:
    """
    根据调整后的表格边界重新组织 grouped_data
    不裁剪图像，只过滤和重新组织数据
    
    Args:
        grouped_data: 原始分组数据
        table_upper_bound: 表格区域的上边界
        table_lower_bound: 表格区域的下边界
        region_name: 区域名称 ('table' 或 'table2')
        
    Returns:
        (reorganized_grouped_data, reorganized_ori_grouped)
        - reorganized_grouped_data: 重新组织后的分组数据（元数据格式）
        - reorganized_ori_grouped: 重新组织后的原始 OCR 结果
    """
    reorganized = {label: [] for label in grouped_data.keys()}
    
    if region_name not in grouped_data:
        print(f"[DEBUG] {region_name} 不存在于 grouped_data 中")
        return grouped_data, None
    
    # 获取原始的 table 项
    original_table_items = grouped_data[region_name]
    
    if not original_table_items:
        print(f"[DEBUG] {region_name} 为空")
        return grouped_data, None
    
    # 过滤：只保留在边界内的项
    filtered_items = []
    out_of_bounds_items = []
    
    for item in original_table_items:
        # 获取项的 y 坐标范围
        item_box = polygon_to_bbox(item['box'])
        item_y_min = item_box[1]
        item_y_max = item_box[3]
        
        # 检查项是否完全在边界内
        if item_y_min >= table_upper_bound and item_y_max <= table_lower_bound:
            filtered_items.append(item)
        else:
            out_of_bounds_items.append(item)
    
    # 更新 grouped_data
    grouped_data[region_name] = filtered_items
    
    # 重新组织到其他分组（越界的项重新分配到 'others'）
    if out_of_bounds_items:
        if 'others' not in grouped_data:
            grouped_data['others'] = []
        grouped_data['others'].extend(out_of_bounds_items)
    
    # 输出调试信息
    # print(f"[DEBUG] {region_name} 边界过滤结果:")
    # print(f"  上边界: {table_upper_bound:.1f}")
    # print(f"  下边界: {table_lower_bound:.1f}")
    # print(f"  原始项数: {len(original_table_items)}")
    # print(f"  边界内项数: {len(filtered_items)}")
    # print(f"  越界项数: {len(out_of_bounds_items)}")
    
    return grouped_data, None


def point_trans_dt(w: int, h: int, dt_boxes: np.ndarray, expand: int = 0, width_ex: int = 0) -> np.ndarray:
    """
    将 OCR 检测框坐标从原始坐标系转换为裁剪后的坐标系
    
    Args:
        w: 图像宽度
        h: 图像高度
        dt_boxes: 检测框列表，格式为 [[[x1, y1], [x2, y2], ...], ...]
        expand: 纵向扩展像素数
        width_ex: 横向扩展像素数
        
    Returns:
        转换后的框数组 [[x_min, y_min, x_max, y_max], ...]
    """
    r_boxes = []
    for box in dt_boxes:
        box = np.array(box)
        x_min = max(0, box[:, 0].min() - width_ex)
        x_max = min(w, box[:, 0].max() + width_ex)
        y_min = max(0, box[:, 1].min() - expand)
        y_max = min(h, box[:, 1].max() + expand)
        r_boxes.append([x_min, y_min, x_max, y_max])
    return np.array(r_boxes)


# ==================== Step 0: 图像归一化 ====================

def normalize_image_size(image: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Step0: 图像归一化大小
    
    根据 img.shape 判断横向还是纵向:
    - 横向 (height < width): 归一化为 1100x1560
    - 纵向 (height >= width): 归一化为 2200x1480
    
    Args:
        image: OpenCV 读取的图像数组 (BGR)
        verbose: 是否打印调试信息
        
    Returns:
        (normalized_image, scale_info)
        - normalized_image: 归一化后的图像数组
        - scale_info: 缩放信息 {'orig_h', 'orig_w', 'norm_h', 'norm_w', 'scale_x', 'scale_y'}
    """
    h, w = image.shape[:2]
    
    if h < w:  # 横向
        target_size = (1560, 1100)  # (width, height)
        if verbose:
            print(f"横向图像: {h}x{w} -> {target_size[1]}x{target_size[0]}")
    else:  # 纵向
        target_size = (1480, 2200)  # (width, height)
        if verbose:
            print(f"纵向图像: {h}x{w} -> {target_size[1]}x{target_size[0]}")
    
    normalized = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 计算缩放信息
    scale_info = {
        'orig_h': h,
        'orig_w': w,
        'norm_h': target_size[1],
        'norm_w': target_size[0],
        'scale_x': target_size[0] / w,  # 新宽 / 原宽
        'scale_y': target_size[1] / h   # 新高 / 原高
    }
    
    return normalized, scale_info


# ==================== 可视化函数 ====================

def visualize_ocr_results(image, ocr_results, output_path='./results/detres.jpg'):
    """
    在图像上可视化 OCR 检测结果
    
    Args:
        image: 输入图像 (numpy array, BGR format)
        ocr_results: OCR 检测结果 [[[pts], (text, score)], ...]
        output_path: 输出图像路径
    """
    import os
    
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 复制图像以避免修改原图
    vis_image = image.copy()
    
    # 为每个 OCR 结果绘制边界框
    for det in ocr_results:
        if len(det) < 2:
            continue
        
        ocr_poly = det[0]  # 多边形坐标
        text, score = det[1]  # 文本和置信度
        
        # 将多边形坐标转换为整数
        poly_points = np.array(ocr_poly, dtype=np.int32)
        
        # 绘制多边形
        cv2.polylines(vis_image, [poly_points], True, (0, 255, 0), 2)
        
        # 绘制文本标签
        if len(poly_points) > 0:
            x_min = int(np.min(poly_points[:, 0]))
            y_min = int(np.min(poly_points[:, 1]))
            
            # 绘制文本
            text_label = f"{text}({score:.2f})"
            cv2.putText(vis_image, text_label, (x_min, max(y_min - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 保存图像
    cv2.imwrite(output_path, vis_image)
    print(f"✓ OCR 可视化结果已保存到: {output_path}")
    
    return vis_image

#画det结果
def draw_boxes(image, boxes, scores=None, drop_score=0.5):
    image2=image.copy()
    if scores is None:
        scores = [1] * len(boxes)
    for (box, score) in zip(boxes, scores):
        
        if score < drop_score:
            continue
        box = np.reshape(np.array(box), [-1, 1, 2]).astype(np.int64)
        # print([box])
        image2 = cv2.polylines(np.array(image2), [box], True, (255, 225, 0), 2)
    return image2
# ==================== Layout 分析 ====================

def map_layout_boxes(layout_boxes: np.ndarray, scale_info: Dict) -> np.ndarray:
    """
    将 layout 检测框坐标从原始图像映射到归一化图像
    
    Args:
        layout_boxes: Layout 检测框数组，格式 [[cls_id, score, x1, y1, x2, y2], ...]
        scale_info: 缩放信息 {'scale_x', 'scale_y', ...}
        
    Returns:
        映射后的框数组，坐标对应归一化后的图像
    """
    if len(layout_boxes) == 0:
        return layout_boxes
    
    mapped_boxes = []
    scale_x = scale_info['scale_x']
    scale_y = scale_info['scale_y']
    
    for box in layout_boxes:
        cls_id, score, x1, y1, x2, y2 = box
        # 等比缩放坐标
        mapped_box = [
            cls_id,
            score,
            x1 * scale_x,
            y1 * scale_y,
            x2 * scale_x,
            y2 * scale_y
        ]
        mapped_boxes.append(mapped_box)
    
    return np.array(mapped_boxes)


def nms_keep_highscore(boxes: np.ndarray, iou_thresh: float = 0.7) -> np.ndarray:
    """
    对 boxes 做 NMS，保留 score 高者
    
    Args:
        boxes: 检测框数组，每行格式 [score, x1, y1, x2, y2, ...] 或 [cls_id, score, x1, y1, x2, y2, ...]
        iou_thresh: IoU 阈值
        
    Returns:
        NMS 处理后的框数组
    """
    if len(boxes) == 0:
        return boxes

    # 按 score 从高到低排序 (第二列是score)
    boxes = boxes[boxes[:, 1].argsort()[::-1]]

    keep = []
    while len(boxes) > 0:
        best = boxes[0]
        keep.append(best)
        rest = []
        for b in boxes[1:]:
            if iou_xyxy(best[2:], b[2:]) <= iou_thresh:
                rest.append(b)
        boxes = np.array(rest) if rest else np.array([])
    
    return np.array(keep)


def iou_xyxy(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两矩形的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

#'/data/tensorflow/kath/OCR/models/Tax-layout/trained-yolo2-2026/best_model/infer/yolo3_2016'
def analyze_layout(image_path: str, 
                   model_dir: str = '/data/tensorflow/kath/OCR/models/Tax-layout/trained-pico-2026/infer/picodet_2026',
                   threshold: float = 0.5) -> Tuple[Dict, List, np.ndarray]:
    """
    版面分析: 使用 PaddleDetection 检测器分析图像布局
    
    Args:
        image_path: 图像路径
        model_dir: 模型目录
        threshold: 检测阈值
        
    Returns:
        (result_boxes, labels, filtered_boxes)
        - result_boxes: 原始检测结果
        - labels: 类别标签列表
        - filtered_boxes: 经过 NMS 处理的框
    """
    try:
        # 使用 PaddleDetection 的 Detector 类
        sys.path.insert(0, '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/deploy/python')
        from infer import Detector
        
        # 创建检测器
        detector = Detector(
            model_dir=model_dir,
            device='GPU',
            threshold=threshold
        )
        
        # 执行预测
        result_list = detector.predict_image(
            image_list=[image_path],
            run_benchmark=False,
            visual=False,
            save_results=False
        )
        
        # 从结果中提取 boxes
        
        if result_list and len(result_list) > 0:
            result_dict = result_list[0] if isinstance(result_list, list) else result_list
            boxes = np.array(result_dict.get('boxes', []))
        else:
            print("Layout 分析: 无法解析检测结果")
            return {}, [], np.array([])
        
        # 默认标签 (根据模型配置)
        labels = detector.pred_config.labels if hasattr(detector, 'pred_config') else \
                 ['ink_F', 'ink_T',  'table2',  'table']
        
        # 过滤并进行 NMS
        if len(boxes) > 0:
            boxes = np.array([i for i in boxes if float(i[1]) > threshold])
            
            if len(boxes) > 0:
                boxes = nms_keep_highscore(boxes, iou_thresh=0.6)
        rec_labels = []
        for subbox in boxes:
            cls_id = int(subbox[0])
            rec_labels.append(labels[cls_id])
        print(f"Layout: 检测到 {len(boxes)} 个区域")
        print('结果rec_labels', rec_labels)
        return result_dict, labels, boxes, rec_labels
    
    except Exception as e:
        print(f"Layout 分析失败: {e}")
        import traceback
        traceback.print_exc()
        return {}, [], np.array([])


# ==================== OCR 检测识别 ====================

def init_ocr_model(det_model_dir: str = '/data/tensorflow/kath/OCR/models/ch_PP-OCRv4_det_infer',
                   rec_model_dir: str = '/data/tensorflow/kath/OCR/models/ch_PP-OCRv4_rec_infer') -> Optional['PaddleOCR']:
    """
    初始化 PaddleOCR 模型
    
    Returns:
        PaddleOCR 模型实例，如果导入失败返回 None
    """
    try:
        from paddleocr import PaddleOCR
        ocr = PaddleOCR(
            use_angle_cls=False,
            lang="ch",
            det_model_dir=det_model_dir,
            rec_model_dir=rec_model_dir,
            det_db_thresh=0.4,
            det_algorithm='DB',
            det_db_box_thresh=0.5,
            det_db_unclip_ratio=1.1,
            use_dilation=False
        )
        return ocr
    except ImportError as e:
        print(f"错误: 无法导入 PaddleOCR - {e}")
        return None


def detect_and_recognize(image: np.ndarray, 
                        ocr_model,  # Type: PaddleOCR or None
                        apply_redink_removal: bool = True) -> List:
    """
    OCR 检测识别
    
    Args:
        image: 图像数组
        ocr_model: OCR 模型实例
        apply_redink_removal: 是否进行红墨水去除预处理
        
    Returns:
        OCR 检测结果列表 [[[pts], (text, score)], ...]
    """
    if ocr_model is None:
        print("错误: OCR 模型未初始化")
        return []
    
    try:
        # 尝试应用红墨水去除预处理
        if apply_redink_removal:
            try:
                image = redink_remover(image)
            except Exception as pre_err:
                print(f"警告: 红墨水去除失败 ({pre_err})，使用原图继续")
                # 继续使用原图，不中断流程
        
        result = ocr_model.ocr(image, cls=True)
        return result[0] if result and result[0] else []
    except Exception as e:
        print(f"OCR 识别失败: {e}")
        import traceback
        print(traceback.print_exc())
        return []


def polygon_to_bbox(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    """将 OCR 四点框转换为 (x1, y1, x2, y2) 矩形包围框"""
    poly = np.array(poly)
    x1, y1 = np.min(poly[:, 0]), np.min(poly[:, 1])
    x2, y2 = np.max(poly[:, 0]), np.max(poly[:, 1])
    return x1, y1, x2, y2


def bbox_iou(boxA: Tuple, boxB: Tuple) -> float:
    """计算两个矩形框的IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    
    if union <= 0:
        return 0.0
    return interArea / min(boxAArea, boxBArea)


# ==================== 结果分组 ====================

def find_rightmost_adjacent_texts(target_box: Tuple[float, float, float, float], 
                                  all_items: List[Dict],
                                  image_width: float,
                                  image_height: float,
                                  count: int = 1,
                                  unlimited_horizontal: bool = False) -> List[str]:
    """
    在右侧找最近的相邻文本项（几何规则版）
    
    从target_box本身开始，向右搜索，返回结果包含自己本身（在最前面）
    搜索范围：
    - 水平：向右查找，默认最大距离为图像宽度的 10%；若 unlimited_horizontal=True 则无限制
    - 垂直：±height*1.5 的范围（中心点上下各延伸 1.5 倍高度）
    
    Args:
        target_box: 目标框 (x1, y1, x2, y2)
        all_items: 所有 OCR 检测项
        image_width: 图像宽度
        image_height: 图像高度
        count: 要找的项数
        unlimited_horizontal: 是否无限制水平搜索（用于金额合计等）
        
    Returns:
        找到的文本列表，包含自己本身（在最前面）
    """
    # print('target_box',target_box)
    target_height = target_box[3] - target_box[1]
    max_h_dist = float('inf') if unlimited_horizontal else (image_width * 0.1)
    v_tolerance = target_height * 1.5
    
    candidates = []
    target_y_center = (target_box[1] + target_box[3]) / 2
    # print('v_tolerance', v_tolerance, 'max_h_dist', max_h_dist)
    
    # 首先找到自己本身
    target_self = None
    for item in all_items:
        item_box = polygon_to_bbox(item['box'])
        # 判断是否是自己（box 几乎完全相同）
        if (abs(item_box[0] - target_box[0]) < 1 and 
            abs(item_box[1] - target_box[1]) < 1 and 
            abs(item_box[2] - target_box[2]) < 1 and 
            abs(item_box[3] - target_box[3]) < 1):
            target_self = item
            # print(f'  ✓ 找到自己: {item.get("text", "")}')
            break
    
    # 搜索右侧相邻的文本
    for item in all_items:
        item_box = polygon_to_bbox(item['box'])
        
        # 检查是否在右侧（x1 > target_box[2]，即左边界超过 target 的右边界）
        if item_box[0] >= target_box[2]:
            item_y_center = (item_box[1] + item_box[3]) / 2
            v_dist = abs(item_y_center - target_y_center)
            # print('          item_box:', item['text'], v_dist, item_box[0] - target_box[2])
            
            # 检查垂直距离是否在范围内
            if v_dist <= v_tolerance:
                if '备注' in item['text']:
                    continue
                # print('             *************【temp】',item['text'], v_dist, '符合垂直距离要求, 水平距离:',item_box[0] - target_box[2],'----',max_h_dist)
                h_dist = item_box[0] - target_box[2]
                # 检查水平距离是否在范围内
                if h_dist <= max_h_dist:
                    candidates.append({
                        'text': item.get('text', ''),
                        'h_dist': h_dist,
                        'v_dist': v_dist,
                        'item': item
                    })
    
    # 按水平距离从小到大排序，然后按垂直距离排序
    # candidates.sort(key=lambda x: (x['h_dist'], x['v_dist']))
    
    # 构建结果：自己本身 + 右侧相邻的
    result_texts = []
    
    # 先加入自己本身
    if target_self:
        result_texts.append(target_self.get('text', ''))
        
    
    # 再加入右侧相邻的
    for c in candidates[:count-1]:  # count-1 是因为已经包含了自己
        result_texts.append(c['text'])
    
    if unlimited_horizontal:
        return result_texts
    else:
        return ''.join(result_texts)




def find_below_text(target_box: Tuple[float, float, float, float],
                    all_items: List[Dict],
                    image_width: float,
                    image_height: float) -> str:
    """
    在下方找最近的相邻文本项（几何规则版）
    从target_box的下方开始向下搜索
    搜索范围：垂直=1.5倍高度，水平=±宽度10%
    """
    target_height = target_box[3] - target_box[1]
    max_v_dist = target_height * 1.5
    h_tolerance = image_width * 0.1
    
    candidates = []
    target_x_center = (target_box[0] + target_box[2]) / 2
    
    for item in all_items:
        item_box = polygon_to_bbox(item['box'])
        
        # 检查是否在下方
        if item_box[1] > target_box[3]:
            item_x_center = (item_box[0] + item_box[2]) / 2
            h_dist = abs(item_x_center - target_x_center)
            
            # 检查水平距离是否在范围内
            if h_dist <= h_tolerance:
                v_dist = item_box[1] - target_box[3]
                # 检查垂直距离是否在范围内
                if v_dist <= max_v_dist:
                    candidates.append({
                        'text': item.get('text', ''),
                        'v_dist': v_dist,
                        'h_dist': h_dist
                    })
    
    if candidates:
        # candidates.sort(key=lambda x: (x['v_dist'], x['h_dist']))
        return candidates[0]['text']
    
    return ''


def extract_structured_fields(grouped_data: Dict,
                              ocr_results: List,
                              scale_info: Dict,
                              is_vertical: bool) -> Tuple[Dict, Dict, Optional[float], Optional[float]]:
    """
    从 OCR 结果中提取结构化字段
    
    Args:
        grouped_data: 按 layout 分组的数据
        ocr_results: 原始 OCR 检测结果（未分组）
        scale_info: 图像缩放信息（包含 norm_h, norm_w）
        is_vertical: 是否为竖版（h >= w）
        
    Returns:
        (extracted_fields, grouped_data_updated, table_upper_bound, table_lower_bound)
        - extracted_fields: 提取的字段字典
        - grouped_data_updated: 修改后的分组数据（table 边界已调整）
        - table_upper_bound: table 的上边界（如果已计算）
        - table_lower_bound: table 的下边界（如果已计算）
    """
    # 构建所有项的映射（用于查找）
    all_items_list = []
    for det in ocr_results:
        all_items_list.append({
            'text': det[1][0],
            'score': det[1][1],
            'box': det[0]
        })
    
    # 获取图像尺寸（用于计算搜索范围）
    image_width = scale_info.get('norm_w', 1480)
    image_height = scale_info.get('norm_h', 2200)
    
    extracted = {}
    extracted['页码'] = {}
    # 初始化 table_upper_bound 和 table_lower_bound
    # 竖版：初始化为 table2 的边界
    # 横版：初始化为 table 的边界
    table_upper_bound = None
    table_lower_bound = None
    
    if is_vertical:
        # 竖版：从 table2 项中获取上下边界
        table2_items = grouped_data.get('table2', [])
        if table2_items:
            
            table_upper_bound = min([min([pt[1] for pt in item['box']]) for item in table2_items])
            table_lower_bound = max([max([pt[1] for pt in item['box']]) for item in table2_items])
    else:
        # 横版：从 table 项中获取上下边界
        table_items = grouped_data.get('table', [])
        if table_items:
            # print('【DDDEEEBBBUUUGGG】',[item['box'] for item in table_items])
            table_upper_bound = min([min([pt[1] for pt in item['box']]) for item in table_items])
            table_lower_bound = max([max([pt[1] for pt in item['box']]) for item in table_items])
            # print('【DDDEEEBBBUUUGGG】',table_upper_bound, table_lower_bound)
    if is_vertical:
        # 竖版：一定要包含的字段
        table2_items = grouped_data.get('table2', [])
        others_items = grouped_data.get('others', [])
        all_items_for_search = others_items + table2_items if table2_items else others_items

        # 初始化字段
        extracted['税务机关'] = ''
        extracted['填发日期'] = ''
        extracted['纳税人名称'] = ''
        extracted['纳税人识别号'] = ''
        extracted['金额合计'] = ['', '']
        extracted['填票人'] = ''
        extracted['编号'] = ''

        date_pattern = re.compile(r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?')

        # 定义要查找的字段标签
        target_fields = ['税务机关', '填发日期', '纳税人名称', '纳税人识别号']

        # 遍历所有det框，查找字段标签并提取对应值
        # print("[DEBUG] 提取税务机关/填发日期/纳税人名称/纳税人识别号（竖版）...")
        for item in all_items_for_search:
            if not isinstance(item['text'], str):
                continue

            text = item['text']
            text_cn = re.sub(r'[^\u4e00-\u9fff]', '', text)

            matched_field = None
            for targetitem in target_fields:
                if similarity(text_cn, targetitem) >= 0.75 or targetitem in text_cn[:7]:
                    matched_field = targetitem
                    break

            if matched_field is None:
                continue

            target_box = polygon_to_bbox(item['box'])
            print(f'\n   *** {text}')
            adj_texts = find_rightmost_adjacent_texts(
                target_box,
                all_items_for_search,
                image_width,
                image_height,
                count=4
            )

            if adj_texts:
                if matched_field == '填发日期':
                    m = date_pattern.search(adj_texts)
                    extracted[matched_field] = m.group() if m else re.sub(matched_field, '', adj_texts)
                else:
                    extracted[matched_field] = re.sub(r'[:：|]|' + matched_field, '', adj_texts)
                print(f"  ✓ {matched_field}: {extracted[matched_field]}")

        # 如果填发日期还是空，在所有项中直接匹配日期格式
        if not extracted['填发日期']:
            for item in all_items_for_search:
                if isinstance(item['text'], str):
                    match = date_pattern.search(item['text'])
                    if match:
                        extracted['填发日期'] = match.group()
                        print(f"  ✓ 填发日期（直接匹配）: {extracted['填发日期']}")
                        break

        # 金额合计：取右侧所有符合条件的文本（无限制水平距离）
        # print("[DEBUG] 提取金额合计（竖版）...")
        for item in all_items_for_search:
            text_cn = re.sub(r'[^\u4e00-\u9fff]', '', item['text'])
            if '金额合计' in text_cn[:5] or similarity(text_cn, '金额合计') >= 0.75:
                print(f'\n   *** {item["text"]}')
                target_box = polygon_to_bbox(item['box'])
                adj_texts = find_rightmost_adjacent_texts(
                    target_box,
                    all_items_for_search,
                    image_width,
                    image_height,
                    count=10,
                    unlimited_horizontal=True
                )
                if adj_texts:
                    extracted['金额合计'] = adj_texts[1:3]
                    print(f"  ✓ 金额合计: {extracted['金额合计']}")
                break

        # 纳税人识别号 fallback：处理 OCR 将名称与标签合并在同一框的情况
        # 例如: '格力数字科技（河北）有限公司纳税人识别号' → 识别号在其右侧
        if not extracted['纳税人识别号']:
            for item in all_items_for_search:
                if isinstance(item['text'], str) and '纳税人识别号' in item['text'] and len(item['text']) > len('纳税人识别号'):
                    target_box = polygon_to_bbox(item['box'])
                    adj_list = find_rightmost_adjacent_texts(
                        target_box, all_items_for_search, image_width, image_height,
                        count=2, unlimited_horizontal=True
                    )
                    if len(adj_list) >= 2:
                        extracted['纳税人识别号'] = adj_list[1]
                        print(f"  ✓ 纳税人识别号 (merged fallback): {extracted['纳税人识别号']}")
                    # 修正纳税人名称末尾残留的'纳税人识别号'
                    if extracted['纳税人名称'].endswith('纳税人识别号'):
                        extracted['纳税人名称'] = extracted['纳税人名称'][:-len('纳税人识别号')].strip()
                        print(f"  ✓ 纳税人名称 (已修正末尾): {extracted['纳税人名称']}")
                    break

        # 金额合计 fallback：标签未被 OCR 识别时，直接通过大写金额/数字金额模式匹配
        if extracted['金额合计'] == ['', '']:
            print("[DEBUG] 金额合计标签未找到，尝试模式匹配...")
            daXie_val = ''
            shuzi_val = ''
            daXie_pattern_big = re.compile(r'[壹贰叁肆伍陆柒捌玖拾佰仟万亿零]+[元角分]')
            jine_pattern_sym = re.compile(r'[¥￥]\d')
            for item in all_items_for_search:
                text = item['text']
                if not daXie_val and daXie_pattern_big.search(text):
                    daXie_val = text
                if not shuzi_val and jine_pattern_sym.search(text):
                    shuzi_val = text
            if daXie_val or shuzi_val:
                extracted['金额合计'] = [daXie_val, shuzi_val]
                print(f"  ✓ 金额合计 (pattern fallback): {extracted['金额合计']}")

        # 编号：找关键字项
        # 编号：格式为 数字+（月份）+数字+证明+数字，如 '26（0122）37证明00005273'
        # 排除纯标题 '税收完税证明' 等（证明前无数字）
        # print("[DEBUG] 提取编号（竖版）...")
        _编号_pattern = re.compile(r'\d+.*证明')
        for item in all_items_for_search:
            if not isinstance(item['text'], str):
                continue
            text = item['text']
            if 'No.' in text or '编号' in text or _编号_pattern.search(text):
                extracted['编号'] = text
                print(f"  ✓ 编号: {extracted['编号']}")
                break
        
        # 金额合计（2项）：找金额关键字项，从它的右边向右搜索（无限制水平距离）
        # print("[DEBUG] 提取金额合计...")
        # extracted['金额合计'] = ['', '']
        # for item in others_items:
        #     # if isinstance(item['text'], str) and any(kw in item['text'] for kw in ['金额合计', '合计', '金额', '¥', '￥', '元']):
        #     target_box = polygon_to_bbox(item['box'])
        #     adj = find_rightmost_adjacent_texts(target_box, all_items_list, image_width, image_height, count=10, unlimited_horizontal=True)
        #     if adj:
        #         extracted['金额合计'] = adj
        #         if len(extracted['金额合计']) < 2:
        #             extracted['金额合计'].extend([''] * (2 - len(extracted['金额合计'])))
        #         break
        
        # 填票人（竖版）：找同一行所有框拼接
        # print("[DEBUG] 提取填票人（竖版）...")
        for item in others_items:
            if isinstance(item['text'], str) and (any(kw in item['text'] for kw in ['填票人']) or similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']), '填票人') >= 0.75):
                target_box = polygon_to_bbox(item['box'])
                y_center = (target_box[1] + target_box[3]) / 2
                row_h = target_box[3] - target_box[1]
                # 找同一行（y中心在1倍行高范围内）且x起点不早于当前框起点的所有项
                row_items = [i for i in others_items
                             if polygon_to_bbox(i['box'])[0] >= target_box[0]
                             and abs((polygon_to_bbox(i['box'])[1] + polygon_to_bbox(i['box'])[3]) / 2 - y_center) <= row_h]
                row_items.sort(key=lambda i: polygon_to_bbox(i['box'])[0])
                combined = ''.join(i['text'] for i in row_items)
                extracted['填票人'] = re.sub(r'[:：|]|(填票人)', '', combined)
                # print(f"  ✓ 填票人: {extracted['填票人']}")
                break
        
        # 页码
        
        for item in others_items :
            if isinstance(item['text'], str) and is_page_string( item['text']) and  extracted['页码']=={}:
                target_box = polygon_to_bbox(item['box'])
                adj = find_rightmost_adjacent_texts(target_box, others_items, image_width, image_height, count=3)
                
                parts = extract_page_info(adj)
                if parts:
                    if len(parts) == 2:
                        try:
                            extracted['页码'] = {"currentPage":int(parts[0].strip()),"totalPage":int(parts[1].strip())}  
                            break
                        except:
                            pass
    
    else:
        table_upper_bound=None
        table_lower_bound=None
        # 横版：一定要包含的字段
        # print("\n[DEBUG] === 横版字段提取 ===")
        
        table_items = grouped_data.get('table', [])
        
        others_items = grouped_data.get('others', [])
        
        # 初始化字段
        extracted['税务机关'] = ''
        extracted['填发日期'] = ''
        extracted['纳税人名称'] = ''
        extracted['纳税人识别号'] = ''
        extracted['金额合计'] = ['', '']
        extracted['填票人'] = ''
        extracted['编号'] = ''
        
        # 税务机关：从 others_items 中找，搜索右侧，count=4
        # 构建所有数据的列表，用于搜索
        all_items_for_search =  others_items + table_items  if table_items else others_items
        # print('all items',[i['text'] for i in all_items_for_search])
        # print("[DEBUG] 提取税务机关...")
        # print('others_items',others_items)
        # print('table_items',table_items)
        for item in all_items_for_search:
            # print('-',item['text'],item['box'])
            if isinstance(item['text'], str) and (similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']) , '税务机关')>=0.75 or '税务机关' in item['text'][:5]) and item['box'][0][1]<500:
                target_box = polygon_to_bbox(item['box'])
                adj_texts = find_rightmost_adjacent_texts(target_box, all_items_for_search, image_width, image_height, count=4)
                if '税务机关' in item['text']:
                    adj_texts = re.sub(r'[:：]','', adj_texts)[4:]
                extracted['税务机关'] = adj_texts
                if adj_texts:
                    print(f"  ✓ 税务机关: {extracted['税务机关']}")
                break
            # elif isinstance(item['text'], str) and '税务机关' in item['text'] and (':' in item['text'] or '：' in item['text']) and item['box'][0][1]<500:
            #     parts = re.split(r'\s*[:：]\s*',item['text'], maxsplit=1)
            #     res=[p.strip() for p in parts] if len(parts) == 2 else [item['text'].strip()]
                
            #     extracted['税务机关'] =  res[1]
        # 填发日期：从所有数据中找，搜索右侧，count=4
        # print("[DEBUG] 提取填发日期...")
        date_pattern = re.compile(r'\d{4}[-年]\d{1,2}[-月]\d{1,2}[日]?')
        for item in all_items_for_search:
            if isinstance(item['text'], str) and  (similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']) , '填发日期')>=0.75 or '填发日期' in item['text'][:5])  :
                target_box = polygon_to_bbox(item['box'])
                adj_texts = find_rightmost_adjacent_texts(target_box, all_items_for_search, image_width, image_height, count=5, unlimited_horizontal=True)

                if adj_texts:
                    temp=''.join(adj_texts)
                    # print("temp:", temp)
                    
                    matched = date_pattern.search(temp)
                    
                    if matched:  
                        # 这里 matched 是 Match 对象，要拿 .group()
                        extracted['填发日期'] = matched.group()
                        # print("全串匹配成功：", extracted['填发日期'])
                    else:
                        # 单独找
                        for t in adj_texts:
                            m = date_pattern.search(t)
                            # print(t, m)
                            if m:
                                extracted['填发日期'] = m.group()
                                # print("单个字段匹配成功：", extracted['填发日期'])
                                break
                        
                        # 如果还没找到
                        if not extracted['填发日期']:
                            extracted['填发日期'] = adj_texts[0]
                            # print("fallback:", adj_texts[0])

        
        if extracted['填发日期']:
            print(f"  ✓ 填发日期: {extracted['填发日期']}")
        
        # 纳税人名称：从 table_items 中找，搜索右侧，count=4
        # print("[DEBUG] 提取纳税人名称...")
        
        for item in all_items_for_search:
            if isinstance(item['text'], str) and  (similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']) , '纳税人名称')>=0.8 or '纳税人名称' in re.sub(r'[^\u4e00-\u9fff]', '', item['text'])[:6]) :
                target_box = polygon_to_bbox(item['box'])
                adj_texts = find_rightmost_adjacent_texts(target_box,all_items_for_search, image_width, image_height, count=4)
                extracted['纳税人名称'] =re.sub(r'[:：|]|(纳税人名称)','',adj_texts) 
                if adj_texts:
                    # print('【DDDEEEBBBUUUGGG】','纳税人名称', item['box'][2][1],table_upper_bound)
                    if table_upper_bound:
                        table_upper_bound=(table_upper_bound+item['box'][2][1])/2
                    else:
                        table_upper_bound=item['box'][2][1]
                    print(f"  ✓ 纳税人名称: {extracted['纳税人名称']}")
                break
        
        # 纳税人识别号：从 table_items 中找，搜索右侧，count=4
        # print("[DEBUG] 提取纳税人识别号...")
        for item in all_items_for_search:
            if isinstance(item['text'], str) and  (similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']) , '纳税人识别号')>=0.8 or '纳税人识别号' in re.sub(r'[^\u4e00-\u9fff]', '', item['text'])[:7]):
                target_box = polygon_to_bbox(item['box'])
                adj_texts = find_rightmost_adjacent_texts(target_box, all_items_for_search, image_width, image_height, count=4)
                extracted['纳税人识别号'] = re.sub(r'[:：|]|(纳税人识别号)','',adj_texts)
                if adj_texts:
                    if adj_texts:
                        # print('【DDDEEEBBBUUUGGG】','纳税人识别号', item['box'][2][1],table_upper_bound)
                        if table_upper_bound:
                            table_upper_bound=(table_upper_bound+item['box'][2][1])/2
                        else:
                            table_upper_bound=item['box'][2][1]
                    print(f"  ✓ 纳税人识别号: {extracted['纳税人识别号']}")
                break
        
        # 金额合计：从 table_items 中找，搜索右侧，count=10，unlimited_horizontal=True
        # print("[DEBUG] 提取金额合计...")
        for item in all_items_for_search:
            if isinstance(item['text'], str) and  (similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']) , '金额合计')>=0.75 or '金额合计' in re.sub(r'[^\u4e00-\u9fff]', '', item['text'])[:5]):
                target_box = polygon_to_bbox(item['box'])
                adj_texts = find_rightmost_adjacent_texts(target_box, all_items_for_search, image_width, image_height, count=10, unlimited_horizontal=True)
                if adj_texts:
                    if  len(adj_texts) >= 3 :
                        extracted['金额合计'] = adj_texts[1:3] 
                    else:
                        extracted['金额合计'] = adj_texts[1:]  + [''] * (3 - len(adj_texts))
                    print(f"  ✓ 金额合计: {extracted['金额合计']}")
                    table_lower_bound=target_box[1]
                break
        
        # 填票人：从 table_items 中找，使用 find_below_text
        # print("[DEBUG] 提取填票人...")
        for item in all_items_for_search:
            if isinstance(item['text'], str) and  (  '填票人' in re.sub(r'[^\u4e00-\u9fff]', '', item['text']) [:4] or similarity(re.sub(r'[^\u4e00-\u9fff]', '', item['text']) , '填票人')>=0.75):
                target_box = polygon_to_bbox(item['box'])
                below = find_below_text(target_box,all_items_for_search, image_width, image_height)
                if below:
                    extracted['填票人'] = below
                    print(f"  ✓ 填票人: {extracted['填票人']}")
                break
        
        # 编号：从 others_items 中找（可选）
        # print("[DEBUG] 提取编号...")
        for item in others_items:
            if isinstance(item['text'], str) and any(kw in item['text'] for kw in ['No.', '编号', '证明']):
                extracted['编号'] = item['text']
                print(f"  ✓ 编号: {extracted['编号']}")
                break
        # 如果table_lower_bound距离其上方还有较大空间，则把table_lower_bound调整为更合理的位置（table_lower_bound之上最后4个det的平均y坐标+这四个det box的平均height的3倍）
        
        if table_upper_bound is not None and table_lower_bound is not None:
            table_upper_bound=table_upper_bound+10
            table_lower_bound=table_lower_bound-10
            # print('upper:',table_upper_bound,'lower:',table_lower_bound)
            # table_upper_bound 和 table_lower_bound 已经在上面计算好了
            # 这里直接用这两个边界重新定义 table，覆盖掉原本提取的 layout 结果
            
            # 过滤 grouped_data['table'] 中在这两个边界内的项
            if 'table' in grouped_data:
                original_table_items = grouped_data['table']
                adjusted_table_items = []
                
                for item in original_table_items:
                    item_box = polygon_to_bbox(item['box'])
                    item_y_min = item_box[1]
                    item_y_max = item_box[3]
                    
                    # 检查项是否在调整后的边界内
                    if item_y_min >= table_upper_bound and item_y_max <= table_lower_bound:
                        adjusted_table_items.append(item)
                
                # 用调整后的项覆盖原本的 table
                grouped_data['table'] = adjusted_table_items
                # print("\n",adjusted_table_items)
                
                # print(f"[DEBUG] table 边界已调整:")
                # print(f"  上边界: {table_upper_bound:.1f}")
                # print(f"  下边界: {table_lower_bound:.1f}")
                # print(f"  调整前项数: {len(original_table_items)}, 调整后项数: {len(adjusted_table_items)}")
                # _,grouped_data=reorganize_grouped_data_by_bounds(grouped_data, table_upper_bound, table_lower_bound,region_name='table' )
        # 把table按照table_upper_bound和table_lower_bound进行重新调整（为以后的table结构识别做准备）    
    
    return extracted, grouped_data, table_upper_bound, table_lower_bound


# ==================== 结果分组 ====================

def group_by_layout_iou(ocr_results: List,
                        layout_boxes: np.ndarray,
                        labels: List[str],
                        iou_thresh: float = 0.5,
                        scale_info: Optional[Dict] = None) -> Tuple[Dict, Dict, Dict]:
    """
    根据 layout 检测结果，使用 IoU>threshold 判定区域归属
    将 OCR 结果按 layout 区域分组
    
    Args:
        ocr_results: OCR检测识别结果 [[[pts], (text, score)], ...]
        layout_boxes: Layout 检测框数组 [[cls_id, score, x1, y1, x2, y2], ...]
        labels: layout 类别名列表
        iou_thresh: IoU 阈值，超过这个值则认为属于该区域
        scale_info: 缩放信息（用于判断竖版还是横版）
        
    Returns:
        (grouped_with_metadata, original_grouped, extracted_fields)
        - grouped_with_metadata: 包含元数据的分组结果
        - original_grouped: 原始 OCR 结果的分组
        - extracted_fields: 提取的结构化字段
    """
    # 根据 '纳税人名称' det框调整 table2 上边界
    # 找到与 '纳税人名称' 文本相似度 >= 0.8 的 det框，将 table2 上边界设为该框下方 5px
    if len(layout_boxes) > 0 and 'table2' in labels and len(ocr_results) > 0:
        table2_idx = labels.index('table2')
        best_match_y2 = None
        best_sim = 0.0
        # 同时收集 税务机关/纳税人名称 的左边界，用于调整 table2 左右边界
        anchor_x1_values = []

        for det in ocr_results:
            text = det[1][0]
            text_cn = re.sub(r'[^\u4e00-\u9fff]', '', text)
            det_box = polygon_to_bbox(det[0])
            # 纳税人名称：用于上边界
            sim = similarity(text_cn, '纳税人名称')
            if sim >= 0.8 and sim > best_sim:
                best_match_y2 = det_box[3]
                best_sim = sim
            # 税务机关 / 纳税人名称：收集左边界
            if similarity(text_cn, '税务机关') >= 0.75 or similarity(text_cn, '纳税人名称') >= 0.8:
                anchor_x1_values.append(det_box[0])

        for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(layout_boxes):
            if int(cls_id) == table2_idx:
                new_y1 = best_match_y2 + 5 if best_match_y2 is not None else y1
                # 左右边界：取两个锚框左边界最小值 - 10
                if anchor_x1_values:
                    img_w = scale_info.get('norm_w', 1480) if scale_info else 1480
                    new_x1 = min(anchor_x1_values) - 10
                    new_x1 = max(0, new_x1)
                    # 右边界与左边界相对图像左右边对称：右边界距图像右边 = 左边界距图像左边
                    new_x2 = img_w - new_x1
                    new_x2 = min(img_w, new_x2)
                    layout_boxes[i] = (cls_id, conf, new_x1, new_y1, new_x2, y2)
                    # print(f"[DEBUG] table2 上边界根据'纳税人名称'调整: y1={y1:.1f} -> {new_y1:.1f}")
                    # print(f"[DEBUG] table2 左右边界调整: x1={x1:.1f} -> {new_x1:.1f}, x2={x2:.1f} -> {new_x2:.1f}")
                else:
                    layout_boxes[i] = (cls_id, conf, x1, new_y1, x2, y2)
                    # print(f"[DEBUG] table2 上边界根据'纳税人名称'调整: y1={y1:.1f} -> {new_y1:.1f}")
                break

    
    grouped = {label: [] for label in labels}
    grouped['others'] = []
    ori_grouped = {label: [] for label in labels}
    ori_grouped['others'] = []
    
    
    for det in ocr_results:
        ocr_poly = det[0]
        text, score = det[1]
        ocr_box = polygon_to_bbox(ocr_poly)
        
        best_iou = 0.0
        best_label = None
        
        # 遍历 layout boxes
        for cls_id, conf, x1, y1, x2, y2 in layout_boxes:
            

            layout_box = (x1, y1, x2, y2)
            iou = bbox_iou(layout_box, ocr_box)
        
            if iou > best_iou:
                best_iou = iou
                best_label = labels[int(cls_id)]
        
        # 创建元数据条目
        entry = {
            "text": text,
            "score": float(score),
            "iou": float(best_iou),
            "box": ocr_poly
        }
        
        # 根据 IoU 阈值分配
        if best_iou > iou_thresh:
            grouped[best_label].append(entry)
            ori_grouped[best_label].append(det)
        else:
            grouped['others'].append(entry)
            ori_grouped['others'].append(det)
    
   
    
    # print('[    debug] groupe[others]:',grouped['others'])
    # 对 table 和 table2 的下边界进行自适应收缩
    # 如果最后4个det的平均纵坐标距离下边界过远（>5倍det高度），则向上收缩/
    # 更新，以下只针对table2，对于table有其他的逻辑
    for label in ['table2']: 
        if label in labels and label in grouped and len(grouped[label]) >= 4:
            label_idx = labels.index(label)
            # 找到对应的 layout box 索引
            layout_idx = None
            for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(layout_boxes):
                if int(cls_id) == label_idx:
                    layout_idx = i
                    break
            
            if layout_idx is not None:
                cls_id, conf, x1, y1, x2, y2 = layout_boxes[layout_idx]
                
                # 取最后4个 det 框
                last_4_items = grouped[label][-4:]
                top_4_items = grouped[label][:4]
                det_heights = []
                det_y_coords = []

                det_heights_top = []
                det_y_coords_top = []

                for item in last_4_items:
                    item_box = polygon_to_bbox(item['box'])
                    det_height = item_box[3] - item_box[1]
                    det_y_center = (item_box[1] + item_box[3]) / 2
                    det_heights.append(det_height)
                    det_y_coords.append(det_y_center)
                
                for item in top_4_items:
                    item_box =item ['box']
                    
                    
                    det_heights_top.append(item_box[0][1])
                    
                
                # 计算平均 det 高度和平均纵坐标
                avg_det_height = np.mean(det_heights)
                avg_det_y = np.mean(det_y_coords)
                
                avg_det_y_top = np.mean(det_heights_top)

                # 计算距离下边界的距离
                dist_to_bottom = y2 - avg_det_y
                
                # 如果距离 > 5倍 det 高度，则向上收缩
                if dist_to_bottom > 5 * avg_det_height:
                    # 收缩到距平均纵坐标 3倍 det 高度的距离
                    y2_shrink = avg_det_y + 2 * avg_det_height
                    layout_boxes[layout_idx] = (cls_id, conf, x1, y1, x2, y2_shrink)
                    
                    # print(f"[DEBUG] {label} 下边界自适应收缩")
                    # print(f"  原下边界: y2={y2:.1f}")
                    # print(f"  新下边界: y2={y2_shrink:.1f}")
                    # print(f"  最后4个det平均纵坐标: {avg_det_y:.1f}, 平均高度: {avg_det_height:.1f}")
                    # print(f"  距离: {dist_to_bottom:.1f} (阈值: {5 * avg_det_height:.1f})")
                    y2=y2_shrink
                # print(f"[DEBUG] {label} 上边界检查")
                # print(f"  原上边界: y1={y1:.1f}","avg_det_y_top :",avg_det_y_top)
                if avg_det_y_top < y1:
                    
                    y1_extended = avg_det_y_top - 10
                    layout_boxes[layout_idx] = (cls_id, conf, x1, y1_extended, x2, y2)
                    # print(f"[DEBUG] {label} 上边界自适应延展")
                    # print(f"  原上边界: y1={y1:.1f}")
                    # print(f"  新上边界: y1={y1_extended:.1f}")
                    # print(f"  前4个det平均纵坐标: {avg_det_y_top:.1f}")
                    y1=y1_extended

    # 对每个分组内的 items 进行排序，保持从上到下、从左到右的顺序
    # for label in grouped:
    #     if len(grouped[label]) > 0:
    #         grouped[label] = sorted(grouped[label], key=lambda item: (
    #             min([pt[1] for pt in item['box']]),  # 从上到下：按 y 坐标最小值排序
    #             min([pt[0] for pt in item['box']])   # 从左到右：按 x 坐标最小值排序
    #         ))
    #         if label == 'table1':
    #             print(f"[DEBUG] table1 分组排序后的顺序:")
    #             for idx, item in enumerate(grouped[label]):
    #                 print(f"  {idx}: {item['text']}")
    
    # for label in ori_grouped:
    #     if len(ori_grouped[label]) > 0:
    #         ori_grouped[label] = sorted(ori_grouped[label], key=lambda det: (
    #             min([pt[1] for pt in det[0]]),  # 从上到下：按 y 坐标最小值排序
    #             min([pt[0] for pt in det[0]])   # 从左到右：按 x 坐标最小值排序
    #         ))
    
    # 提取结构化字段
    is_vertical = scale_info and scale_info.get('norm_h', 0) >= scale_info.get('norm_w', 0)
    extracted_fields, grouped, table_upper_bound, table_lower_bound = extract_structured_fields(grouped, ocr_results, scale_info, is_vertical)
    # print("\n[DEBUG] 提取的结构化字段:")
    # print(grouped)
    # 如果计算出了 table 的上下边界，则修改对应的 layout_boxes
    if table_upper_bound is not None and table_lower_bound is not None and 'table' in labels:
        table_idx = labels.index('table')
        for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(layout_boxes):
            if int(cls_id) == table_idx:
                # 修改这个 layout box 的上下边界
                layout_boxes[i] = (cls_id, conf, x1, table_upper_bound, x2, table_lower_bound)
                
                # print(f"[DEBUG] layout_boxes 中 table 的边界已更新:")
                # print(f"  原: y1={y1:.1f}, y2={y2:.1f}")
                # print(f"  新: y1={table_upper_bound:.1f}, y2={table_lower_bound:.1f}")
                break
        
        # 新增：在调整后的边界内，检查最下方的 4 个 detbox，并微调 table_lower_bound
        if 'table' in grouped and len(grouped['table']) >= 4:
            # 过滤出在新边界内的所有 table 项
            table_items_in_bounds = []
            for item in grouped['table']:
                item_box = polygon_to_bbox(item['box'])
                item_y_min = item_box[1]
                item_y_max = item_box[3]
                
                if item_y_min >= table_upper_bound and item_y_max <= table_lower_bound:
                    table_items_in_bounds.append(item)
            
            # 如果有足够的项，找最下方的 4 个
            if len(table_items_in_bounds) >= 4:
                # 按下边界 y 坐标排序，取最下方的 4 个
                sorted_items = sorted(table_items_in_bounds, 
                                    key=lambda item: polygon_to_bbox(item['box'])[3],  # 按 y2 排序
                                    reverse=True)  # 降序，最下方的在前
                
                last_4_items = sorted_items[:4]
                
                # 计算这 4 个 box 的下界 y 坐标和平均高度
                y2_coords = []
                heights = []
                
                for item in last_4_items:
                    box = polygon_to_bbox(item['box'])
                    y2_coords.append(box[3])  # 下边界 y 坐标
                    heights.append(box[3] - box[1])  # 高度
                
                avg_y2 = np.mean(y2_coords)
                avg_height = np.mean(heights)
                
                # 计算距离
                distance_to_bound = table_lower_bound - avg_y2
                
                # print(f"[DEBUG] table_lower_bound 微调检查:")
                # print(f"  新边界内的 table 项数: {len(table_items_in_bounds)}")
                # print(f"  最下方 4 个 box 的平均下界: {avg_y2:.1f}")
                # print(f"  平均高度: {avg_height:.1f}")
                # print(f"  当前 table_lower_bound: {table_lower_bound:.1f}")
                # print(f"  距离: {distance_to_bound:.1f} (阈值: {5 * avg_height:.1f})")
                
                # 如果距离 > 5 倍平均高度，则向上移动
                if distance_to_bound > 5 * avg_height:
                    new_table_lower_bound = avg_y2 + 2 * avg_height
                    # print(f"  ✓ table_lower_bound 需要向上调整: {table_lower_bound:.1f} -> {new_table_lower_bound:.1f}")
                    
                    # 更新 table_lower_bound
                    table_lower_bound = new_table_lower_bound
                    
                    # 更新 layout_boxes 中的 table 边界
                    for i, (cls_id, conf, x1, y1, x2, y2) in enumerate(layout_boxes):
                        if int(cls_id) == table_idx:
                            layout_boxes[i] = (cls_id, conf, x1, table_upper_bound, x2, table_lower_bound)
                            # print(f"  ✓ layout_boxes 已更新为新的下界: {table_lower_bound:.1f}")
                            break
                    
                    # 重新过滤 grouped['table']，使用新的边界
                    adjusted_table_items = []
                    for item in grouped['table']:
                        item_box = polygon_to_bbox(item['box'])
                        item_y_min = item_box[1]
                        item_y_max = item_box[3]
                        
                        if item_y_min >= table_upper_bound and item_y_max <= table_lower_bound:
                            adjusted_table_items.append(item)
                    
                    grouped['table'] = adjusted_table_items
                    # print(f"  ✓ 重新过滤后 table 项数: {len(adjusted_table_items)}")
    
    
    return grouped, ori_grouped, extracted_fields


# ==================== 表格识别 ====================

class TableRecognizer:
    """表格识别器 - 完全按照 tax_code_imp.ipynb 实现"""
    
    def __init__(self):
        """初始化表格识别器 - 按notebook初始化"""
        self.struct_model = None
        self.table_arg = None
    
    def _init_table_arg(self):
        """初始化 table_arg - 完全按照notebook复制"""
        import argparse
        import os
        
        # 模型路径定义 - 按照notebook
        TABLE_MODEL_DIR = '/data/tensorflow/kath/OCR/models/official/ch_ppstructure_mobile_v2.0_SLANet_infer'
        TABLE_DICT_DIR = '/data/tensorflow/kath/OCR/PaddleOCR-release-2.7/ppocr/utils/dict/table_structure_dict_ch.txt'
        FONT_DIR = '/data/tensorflow/kath/OCR/PaddleDetection-release-2.7/ppdet/utils/simfang.ttf'
        
        # 完全按照notebook创建table_arg - 不做任何改动
        self.table_arg=argparse.Namespace(alpha=1.0, alphacolor=(255, 255, 255), benchmark=False, beta=1.0, binarize=False, cls_batch_num=6, cls_image_shape='3, 48, 192', 
            cls_model_dir=None, cls_thresh=0.9, cpu_threads=10, crop_res_save_dir='./output', det_algorithm='DB', det_box_type='quad', det_db_box_thresh=0.6, 
            det_db_score_mode='fast', det_db_thresh=0.3, det_db_unclip_ratio=1.5, det_east_cover_thresh=0.1, det_east_nms_thresh=0.2, det_east_score_thresh=0.8, 
            det_limit_side_len=960, det_limit_type='max', det_model_dir=None, det_pse_box_thresh=0.85, det_pse_min_area=16, det_pse_scale=1, det_pse_thresh=0, 
            det_sast_nms_thresh=0.2, det_sast_score_thresh=0.5, draw_img_save_dir='./inference_results', drop_score=0.5, e2e_algorithm='PGNet', 
            e2e_char_dict_path='../ppocr/utils/ic15_dict.txt', e2e_limit_side_len=768, e2e_limit_type='max', e2e_model_dir=None, e2e_pgnet_mode='fast', 
            e2e_pgnet_score_thresh=0.5, e2e_pgnet_valid_set='totaltext', enable_mkldnn=False, fourier_degree=5, gpu_id=0, gpu_mem=500, 
            image_dir='/data/tensorflow/kath/OCR/code/ocr/output/zx_0000033_4.jpeg', image_orientation=False, invert=False, ir_optim=True, kie_algorithm='LayoutXLM', 
            label_list=['0', '180'], layout=True, layout_dict_path='../ppocr/utils/dict/layout_dict/layout_publaynet_dict.txt', layout_model_dir=None, layout_nms_threshold=0.5, 
            layout_score_threshold=0.5, max_batch_size=10, max_text_length=25, merge_no_span_structure=True, min_subgraph_size=15, mode='structure', ocr=True, ocr_order_method=None, 
            output='/data/tensorflow/kath/OCR/code/ocr/output/sub/table', page_num=0, precision='fp32', process_id=0, re_model_dir=None, rec_algorithm='SVTR_LCNet', rec_batch_num=6, 
            rec_char_dict_path='./ppocr/utils/ppocr_keys_v1.txt', rec_image_inverse=True, rec_image_shape='3, 48, 320', rec_model_dir=None, recovery=False, save_crop_res=False, 
            save_log_path='./log_output/', scales=[8, 16, 32], ser_dict_path='../train_data/XFUND/class_list_xfun.txt', ser_model_dir=None, show_log=True, sr_batch_num=1, 
            sr_image_shape='3, 32, 128', sr_model_dir=None, table=True, table_algorithm='TableAttn', table_char_dict_path=os.path.dirname(TABLE_DICT_DIR)+'/table_structure_dict_ch.txt', 
            table_max_len=488, table_model_dir=TABLE_MODEL_DIR, total_process_num=1, use_angle_cls=False, use_dilation=False, use_gpu=True, 
            use_mp=False, use_npu=False, use_onnx=False, use_pdf2docx_api=False, use_pdserving=False, 
            use_space_char=True, use_tensorrt=False, use_visual_backbone=True, use_xpu=False, vis_font_path=FONT_DIR+'simfang.ttf', warmup=False)
    
    def recognize_table(self, cropedimg, croped_target_det, rec_res):
        """
        识别表格 - 原封不动从notebook复制
        参数:
            cropedimg: 裁剪后的表格图像 (来自 crop_image_from_boxrow)
            croped_target_det: 裁剪后的检测结果
            target_det: 原始检测结果
        返回:
            (data, res) - DataFrame 和 HTML 字符串
        """
        try:
            # 初始化模型（如果还没有）
            if self.struct_model is None:
                self._init_struct_model()
            
            if self.struct_model is None:
                print("    ! 结构模型初始化失败")
                return None, None
            
            # 完全按照notebook Cell 74复制：
            # dt_boxes=[i[0] for i in croped_target_det]
            # print(' croped_target_det', croped_target_det)
            # dt_boxes=[i[0] for i in croped_target_det]
            dt_boxes=croped_target_det
            # print('dt_boxes',dt_boxes)
            h,w = cropedimg.shape[:2] 
            
            # rec_res=[i[1] for i in target_det]
            # rec_res=[i[1] for i in target_det]
            
            # structure = structure_rec(cropedimg,struct_model)
            # cv2.imwrite('/data/tensorflow/kath/Service/TaxParser/results/cropedimg.jpg',cropedimg)
            structure = self.structure_rec(cropedimg)
            html_tokens, bboxes= structure
            # visualize_table_annotation(cropedimg, bboxes,html_tokens, score=None)
            # cv2.imwrite('/data/tensorflow/kath/Service/TaxParser/results/tax_table_dtboxes_debug.jpg',cropedimg)
           
            # imggg=draw_boxes(cropedimg, dt_boxes)
            # cv2.imwrite('/data/tensorflow/kath/Service/TaxParser/results/tax_det_dtboxes_debug.jpg',imggg)
            # structure = structure_box_trans(structure,0,0,w,h)
            structure = self.structure_box_trans(structure,0,0,w,h)
            
            # from ppstructure.table.matcher import TableMatch as TM
            import sys
            ppocr_root = '/data/tensorflow/kath/OCR/PaddleOCR-release-2.7'
            if ppocr_root not in sys.path:
                sys.path.insert(0, ppocr_root)
            
            from ppstructure.table.matcher import TableMatch as TM
            match1=TM()
            
            # res=match1(structure,dt_boxes, rec_res)
            
            
            res=match1(structure,dt_boxes, rec_res) 
            
            # data = pd.read_html(res, header=None, keep_default_na=False)[0]
            data = pd.read_html(res, header=None, keep_default_na=False)[0]
            # print('\n\n\n【ORIGINAL HTML】',html_tokens)
            print('gen-dataframe:',data)
            # print('\nres::: ',res)
            return data, res
        
        except Exception as e:
            print(f"  ! 表格识别失败: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def structure_rec(self, img):
        """从notebook复制: def structure_rec(img,model):"""
        structure_res, elapse = self.struct_model(img)
        return structure_res
    
    def structure_box_trans(self, structure, ww, hh, w, h):
        """从notebook的imagetools.py复制的坐标变换函数"""
        _, boxes = structure
        firstbox = 1
        firstbox_ = []
        for i in boxes:
            if firstbox == 0:
                firstbox_ = i
            i[0] = max(0, i[0] - ww)
            i[1] = max(0, i[1] - hh)
            i[2] = min(w, i[2] + ww)
            i[3] = max(0, i[3] - hh)
            i[4] = min(w, i[4] + ww)
            i[5] = min(h, i[5] + hh)
            i[6] = max(0, i[6] - ww)
            i[7] = min(h, i[7] + hh)
        return _, boxes
    
    def _init_struct_model(self):
        """初始化模型 - 完全按照notebook的Struct_model()函数复制"""
        print('  loading Structure model...')
        try:
            # 完全按照notebook:
            
            
            # 初始化table_arg
            if self.table_arg is None:
                self._init_table_arg()
            
            # def Struct_model():
            #     print('loading Structure model...')
            #     table_structurer = TableStructurer(table_arg)
            #     return table_structurer
            # struct_model=Struct_model()
            
            # 关键：notebook中的TableStructurer就是PPStructure!
            import inspect
            print(inspect.getfile(TableStructurer))
            self.struct_model = TableStructurer(self.table_arg)
            
            print('  ✓ Structure model loaded')
        
        except Exception as e:
            print(f"  ! 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self.struct_model = None


# ==================== 数据输出 ====================

class ResultProcessor:
    """结果处理和输出"""
    
    def __init__(self, output_dir: str = './tax_parser_output'):
        """初始化结果处理器"""
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def save_grouped_results(self, grouped_data: Dict, labels: List[str],
                            filename: str = 'grouped_results.json') -> str:
        """
        保存分组结果为 JSON
        
        Args:
            grouped_data: 分组数据
            labels: 标签列表
            filename: 输出文件名
            
        Returns:
            输出文件路径
        """
        output_path = os.path.join(self.output_dir, filename)
        
        # 转换为可序列化格式
        serializable_data = {}
        for key, items in grouped_data.items():
            serializable_data[key] = []
            for item in items:
                if isinstance(item, dict):
                    # 转换 box 到列表格式
                    item_copy = item.copy()
                    if isinstance(item_copy.get('box'), np.ndarray):
                        item_copy['box'] = item_copy['box'].tolist()
                    serializable_data[key].append(item_copy)
                else:
                    # 原始 OCR 结果
                    serializable_data[key].append({
                        'text': item[1][0],
                        'score': float(item[1][1]),
                        'box': [p.tolist() if isinstance(p, np.ndarray) else p for p in item[0]]
                    })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"分组结果已保存: {output_path}")
        return output_path
    
    def save_table_results(self, dataframe: Optional[pd.DataFrame],
                          html_string: Optional[str],
                          base_filename: str = 'table') -> Dict[str, str]:
        """
        保存表格结果（DataFrame CSV + HTML）
        
        Args:
            dataframe: pandas DataFrame
            html_string: HTML 字符串
            base_filename: 基础文件名
            
        Returns:
            {'csv': csv_path, 'html': html_path}
        """
        result_paths = {}
        
        if dataframe is not None:
            csv_path = os.path.join(self.output_dir, f'{base_filename}.csv')
            dataframe.to_csv(csv_path, index=False, encoding='utf-8')
            result_paths['csv'] = csv_path
            print(f"表格 CSV 已保存: {csv_path}")
        
        if html_string is not None:
            html_path = os.path.join(self.output_dir, f'{base_filename}.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_string)
            result_paths['html'] = html_path
            print(f"表格 HTML 已保存: {html_path}")
        
        return result_paths
    
    def save_debug_visualizations(self, 
                                  image: np.ndarray,
                                  layout_boxes: np.ndarray,
                                  labels: List[str],
                                  ocr_boxes: Optional[np.ndarray] = None) -> str:
        """
        保存调试可视化图像
        
        Args:
            image: 原始图像
            layout_boxes: layout 检测框
            labels: 标签列表
            ocr_boxes: OCR 检测框（可选）
            
        Returns:
            保存的图像路径
        """
        img_copy = image.copy()
        colors = {name: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                 for name in labels}
        
        # 绘制 layout 框
        for cls_id, score, x1, y1, x2, y2 in layout_boxes:
            label = labels[int(cls_id)]
            color = colors[label]
            cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(img_copy, f"{label} {score:.2f}",
                       (int(x1), int(y1) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        
        output_path = os.path.join(self.output_dir, 'visualization_layout.jpg')
        # cv2.imwrite(output_path, img_copy)
        print(f"可视化图像已保存: {output_path}")
        
        return output_path


# ==================== 主处理流程 ====================

class TaxDocumentParser:
    """税务文档完整解析器"""
    
    def __init__(self, 
                 output_dir: str = './tax_parser_output',
                 use_gpu: bool = True):
        """
        初始化解析器
        
        Args:
            output_dir: 输出目录
            use_gpu: 是否使用 GPU
        """
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        
        # 初始化各个模块
        self.ocr_model = None
        self.table_recognizer = None
        self.result_processor = ResultProcessor(output_dir)
    
    def _init_models(self):
        """初始化所有模型"""
        if self.ocr_model is None:
            print("初始化 OCR 模型...")
            self.ocr_model = init_ocr_model()
        
        if self.table_recognizer is None:
            print("初始化表格识别器...")
            self.table_recognizer = TableRecognizer()
    
    def _process_table_region(self, region_name: str, 
                             image_normalized: np.ndarray,
                             layout_boxes: np.ndarray,
                             labels: List[str],
                             ori_grouped: Dict,
                             output_dir: Optional[str] = None,
                             crop_counter: Optional[Dict] = None) -> Tuple[Optional[np.ndarray], Optional[List]]:
        """
        处理特定表格区域的裁剪和坐标映射
        
        Args:
            region_name: 区域名称 ('table' 或 'table2')
            image_normalized: 归一化后的图像
            layout_boxes: layout 检测框
            labels: 标签列表
            ori_grouped: 按 layout 分组的原始 OCR 结果
            output_dir: 输出目录（如果提供，将保存裁剪图像）
            crop_counter: 计数器字典，用于生成唯一标识符
            
        Returns:
            (cropped_image, mapped_det_results)
            其中 mapped_det_results 是坐标已映射的 OCR 结果
        """
        try:
            # 从 layout boxes 中找到对应区域的 box
            region_box = None
            region_idx = None
            if region_name in labels:
                region_idx = labels.index(region_name)
                for i, (cls_id, _, _, _, _, _) in enumerate(layout_boxes):
                    if int(cls_id) == region_idx:
                        region_box = layout_boxes[i]
                        break
            
            if region_box is None:
                print(f"  警告: 未找到 {region_name} 的 layout box")
                return None, None
            
            # 使用 det_results 参数进行坐标映射
            # 返回 (cropped_img, mapped_det_results)
            # print('   layout_boxes:', layout_boxes)
            # print(f"  处理区域: {region_name} , region_box: {region_box}")
            cropped_img, mapped_det = crop_image_from_boxrow(
                image_normalized,
                region_box,
                det_results=ori_grouped[region_name]
            )
            
            # 保存裁剪图像（如果提供了输出目录）
            if cropped_img is not None and output_dir is not None and region_name in ['table', 'table2']:
                import os
                os.makedirs(output_dir, exist_ok=True)
                
                # 初始化计数器
                if crop_counter is None:
                    crop_counter = {}
                
                if region_name not in crop_counter:
                    crop_counter[region_name] = 0
                
                # 增加计数并生成 3 位数的唯一标识符
                crop_counter[region_name] += 1
                unique_id = f"{crop_counter[region_name]:03d}"
                
                # 生成文件名
                filename = f"{region_name}_{unique_id}.jpg"
                filepath = os.path.join(output_dir, filename)
                
                # 保存图像
                # cv2.imwrite(filepath, cropped_img)
                # print(f"  ✓ 裁剪图像已保存: {filepath}")
            
            return cropped_img, mapped_det
            
        except Exception as e:
            import traceback
            print(f"  警告: {region_name} 处理失败 ({e})")
            traceback.print_exc()
            return None, None
    
    def process_image(self, image_path: str,
                     save_intermediate: bool = False) -> Dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            save_intermediate: 是否保存中间结果
            
        Returns:
            处理结果字典
        """
        self._init_models()
        
        print(f"\n{'='*60}")
        print(f"处理图像: {image_path}")
        print(f"{'='*60}")
        
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return {'status': 'error', 'message': '无法读取图像'}
        
        # Step 0: 图像归一化
        print("\n[Step 0] 图像归一化...")
        image_normalized, scale_info = normalize_image_size(image, verbose=True)
        # cv2.imwrite('./'+ image_path.split('/')[-1].split('.')[0] +'_image_normalized.jpg', image_normalized)
        # Step 1: Layout 分析
        print("\n[Step 1] Layout 分析...")
        result_layout, labels, layout_boxes,rec_labels = analyze_layout(image_path)
        print('[step 1 debug: labels]',labels)
        
        if layout_boxes is None or len(layout_boxes) == 0:
            print("警告: 未检测到任何布局区域")
            layout_boxes = np.array([])
            labels = []
        
        # Step 1.5: 映射 Layout 框坐标到归一化图像
        if len(layout_boxes) > 0:
            print("\n[Step 1.5] 映射 Layout 坐标...")
            layout_boxes = map_layout_boxes(layout_boxes, scale_info)
            # print(f"✓ 坐标已映射 (scale_x={scale_info['scale_x']:.4f}, scale_y={scale_info['scale_y']:.4f})")
        
        # Step 2: OCR 检测识别
        print("\n[Step 2] OCR 检测识别...")
        ocr_results = detect_and_recognize(image_normalized, self.ocr_model)
        print(f"检测到 {len(ocr_results)} 个文本区域")
        # print('DET res', ocr_results)
        
        # 可视化 OCR 结果
        # try:
        #     visualize_ocr_results(image_normalized, ocr_results, './results/detres.jpg')
        # except Exception as e:
        #     print(f"警告: OCR 可视化失败 - {e}")
        
        # 初始化输出目录和裁剪计数器
        output_dir = './results'
        crop_counter = {}
        
        # Step 3: 按 Layout 分组
        print("\n[Step 3] 按 Layout 分组...")
        grouped_data, ori_grouped, extracted_fields = group_by_layout_iou(
            ocr_results, layout_boxes, labels, scale_info=scale_info
        )
        
        # print('\n\n============================grouped_data1===================\n',[ii['text'] for ii in  grouped_data['others']])
        # layout_boxes_ori=layout_boxes.copy()
        grouped_data2, ori_grouped, extracted_fields = group_by_layout_iou(
            ocr_results, layout_boxes, labels, scale_info=scale_info
        )
        # print('\n\n============================grouped_data2===================\n',[ii['text'] for ii in  grouped_data2['others']])


        # print(f"DEBUG [Step 3]: 分组结果统计:")
        # for label, items in grouped_data.items():
        #     print(f"  {label}: {len(items)} 项")
        
        # print(f"DEBUG [Step 3]: 原始分组键: {list(ori_grouped.keys())}")
        if 'ink_F' in rec_labels:
            print('F 标签,公章有误，函数结束')
            return {
            'status': 'wrong-seal',
            'image_path': image_path,
            'grouped_data': grouped_data,
            'table_results': None,
            'labels': labels,
            'extracted_fields': extracted_fields,  # 使用已提取的字段
            'output_dir': None
        }
        if 'ink_T' not in rec_labels and 'ink_F' not in rec_labels:
            print('无盖章函数结束')
            return {
            'status': 'no-seal',
            'image_path': image_path,
            'grouped_data': grouped_data,
            'table_results': None,
            'labels': labels,
            'extracted_fields': extracted_fields,  # 使用已提取的字段
            'output_dir': None
        }

        # Step 4: 表格识别
        print("\n[Step 4] 表格识别...")
        table_results = {'table': None, 'table2': None}
        
        print(f"DEBUG [Step 4]: ori_grouped 键: {list(ori_grouped.keys())}")
        # for key in ori_grouped:
        #     print(f"DEBUG [Step 4]:   {key}: {len(ori_grouped[key])} 项")
        
        # 检查所有可能的表格标签
        possible_table_labels = ['table', 'table2', 'table1', 'ink_F', 'ink_T']
        # print(f"DEBUG [Step 4]: 检查所有可能的表格标签: {possible_table_labels}")
        # for label in possible_table_labels:
        #     if label in ori_grouped:
        #         print(f"DEBUG [Step 4]:   ✓ {label} 存在，{len(ori_grouped[label])} 项")
        #     else:
        #         print(f"DEBUG [Step 4]:   ✗ {label} 不存在")
        
        if 'table2' in ori_grouped and ori_grouped['table2']:
            print("\n[Step 4.1] 处理 table2 区域...")
            table_crop, table_det = self._process_table_region(
                'table2', image_normalized, layout_boxes, labels, ori_grouped,
                output_dir=output_dir, crop_counter=crop_counter
            )
            
            # print(f"DEBUG: table_crop 是否为 None: {table_crop is None}")
            # print(f"DEBUG: table_det 是否为 None: {table_det is None}")
            
            if table_crop is not None and table_det is not None:
                # table_det 是 [[[pts], (text, score)], ...] 格式的 OCR 结果
                # 需要提取 det_boxes 和 rec_results
                print("    处理检测结果数据...")
                dt_boxes = [det[0] for det in table_det]  # 四点坐标
                h,w = table_crop.shape[:2] 
                dt_boxes = point_trans_dt(w,h,dt_boxes)
                rec_results = [det[1] for det in table_det]  # (text, score) 元组
                
                # print(f"    ✓ 提取 {len(dt_boxes)} 个文本框和识别结果")
                
                # 进行表格识别
                # cv2.imwrite('/data/tensorflow/kath/jupyter/output/debug_table2_crop.jpg', table_crop)
                df, html = self.table_recognizer.recognize_table(table_crop, dt_boxes, rec_results)
                print(f"DEBUG: recognize_table 返回的 df: {df is not None}")
                print(f"DEBUG: recognize_table 返回的 html: {html is not None}")
                if html:
                    
                    print(f"DEBUG: html : {html}")
                table_results['table2'] = {'dataframe': df, 'html': html}
                
                if df is not None:
                    print(f"    ✓ 表格识别成功: {df.shape[0]} 行 × {df.shape[1]} 列")
                else:
                    print(f"    ! 表格识别完成但无法转换为 DataFrame")
        
        elif 'table' in ori_grouped and ori_grouped['table']:
            print("\n[Step 4.2] 处理 table 区域...")
            # print('&&&&&&&&&&&&【DEBUUUUUG】',ori_grouped['table'])
            table_crop, table_det = self._process_table_region(
                'table', image_normalized, layout_boxes, labels, ori_grouped,
                output_dir=output_dir, crop_counter=crop_counter
            )
            
            # print(f"      ***********    DEBUG: table_det: {table_det }")
            
            if table_crop is not None and table_det is not None:
                dt_boxes = [det[0] for det in table_det]
                h,w = table_crop.shape[:2] 
                dt_boxes = point_trans_dt(w,h,dt_boxes)
                rec_results = [det[1] for det in table_det]
                
                print(f"    ✓ 提取 {len(dt_boxes)} 个文本框和识别结果")
                # cv2.imwrite('./results/debug_table_crop.jpg', table_crop)
                # print("      &&&&&&&&&&&&    【DEBUG】送入模型的: dt_boxes:", dt_boxes)
                df, html = self.table_recognizer.recognize_table(table_crop, dt_boxes, rec_results)
                
                if html:
                    
                    print(f"DEBUG: html 前100字符: {html}")
                table_results['table'] = {'dataframe': df, 'html': html}
                
                if df is not None:
                    print(f"    ✓ 表格识别成功: {df.shape[0]} 行 × {df.shape[1]} 列")
                else:
                    print(f"    ! 表格识别完成但无法转换为 DataFrame")
        
        # Step 5: 保存结果
        print("\n[Step 5] 保存结果...")
        
        # 保存分组数据
        self.result_processor.save_grouped_results(grouped_data, labels)
        
        # 保存表格结果
        for table_type, table_data in table_results.items():
            if table_data and (table_data['dataframe'] is not None or table_data['html'] is not None):
                self.result_processor.save_table_results(
                    table_data['dataframe'],
                    table_data['html'],
                    f"{table_type}_result"
                )
        
        # 保存可视化
        if len(layout_boxes) > 0:
            self.result_processor.save_debug_visualizations(
                image_normalized, layout_boxes, labels
            )
        
        result = {
            'status': 'success',
            'image_path': image_path,
            'grouped_data': grouped_data,
            'table_results': table_results,
            'labels': labels,
            'extracted_fields': extracted_fields,  # 新增：结构化字段
            'output_dir': self.output_dir
        }
        
        return result
    
    def process_directory(self, directory: str,
                         recursive: bool = False,
                         save_intermediate: bool = False) -> List[Dict]:
        """
        处理一个目录中的所有图像
        
        Args:
            directory: 目录路径
            recursive: 是否递归处理子目录
            save_intermediate: 是否保存中间结果
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 获取所有图像文件
        if recursive:
            image_files = [str(p) for p in Path(directory).rglob('*')
                          if p.suffix.lower() in image_extensions]
        else:
            image_files = [str(p) for p in Path(directory).glob('*')
                          if p.suffix.lower() in image_extensions]
        
        print(f"找到 {len(image_files)} 个图像文件")
        
        for i, image_path in enumerate(image_files, 1):
            print(f"\n处理进度: {i}/{len(image_files)}")
            result = self.process_image(image_path, save_intermediate)
            results.append(result)
        
        # 汇总报告
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: List[Dict]):
        """打印处理汇总"""
        print(f"\n{'='*60}")
        print("处理完成汇总")
        print(f"{'='*60}")
        print(f"处理文件总数: {len(results)}")
        print(f"成功: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"失败: {sum(1 for r in results if r['status'] == 'error')}")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")


# ==================== 命令行接口 ====================

def main():
    """命令行主函数"""
    parser = argparse.ArgumentParser(
        description='Tax Document Parser - 税务文档完整解析工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 处理单张图像
  python tax_document_parser.py --image path/to/image.jpg
  
  # 处理整个目录
  python tax_document_parser.py --directory path/to/images/
  
  # 递归处理目录
  python tax_document_parser.py --directory path/to/images/ --recursive
  
  # 指定输出目录
  python tax_document_parser.py --image image.jpg --output results/
        """
    )
    
    parser.add_argument(
        '--image',
        type=str,
        help='单张图像路径'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        help='图像目录路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./tax_parser_output',
        help='输出目录 (默认: ./tax_parser_output)'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='递归处理子目录'
    )
    
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='使用 CPU 而不是 GPU'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not args.image and not args.directory:
        parser.print_help()
        print("\n错误: 必须指定 --image 或 --directory")
        sys.exit(1)
    
    if args.image and args.directory:
        print("错误: --image 和 --directory 不能同时指定")
        sys.exit(1)
    
    # 创建解析器
    parser_obj = TaxDocumentParser(
        output_dir=args.output,
        use_gpu=not args.cpu
    )
    
    # 处理输入
    if args.image:
        if not os.path.exists(args.image):
            print(f"错误: 图像文件不存在 {args.image}")
            sys.exit(1)
        result = parser_obj.process_image(args.image)
        print(f"\n处理结果: {result['status']}")
    
    elif args.directory:
        if not os.path.isdir(args.directory):
            print(f"错误: 目录不存在 {args.directory}")
            sys.exit(1)
        results = parser_obj.process_directory(
            args.directory,
            recursive=args.recursive
        )
        print(f"处理完成: {len(results)} 个文件")


if __name__ == '__main__':
    main()
