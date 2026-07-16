"""Utility to crop 'table' and 'table2' boxes from an image using detector output.

Expect box format: a dict like {'boxes': np.ndarray([...], dtype=float32), 'boxes_num': np.ndarray([N], dtype=int32)}
Where each row of `boxes` is: [class_id, score, x_min, y_min, x_max, y_max]
`labels` is a list mapping class_id -> class_name, e.g. ['ink_F', 'ink_T', 'table1', 'table2', 'title', 'table']

Example usage:
    from crop_tables import crop_tables_from_boxres
    crops = crop_tables_from_boxres(image_path, boxres, labels, output_dir='crops')

Returns list of dicts: [{'label': 'table', 'score': 0.99, 'box': [x1,y1,x2,y2], 'crop': np.ndarray, 'save_path': '...'}, ...]

"""
from typing import List, Tuple, Dict, Union, Optional
import os
import cv2
import numpy as np
import runpy
import json
from pathlib import Path


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _box_row_to_xyxy(row: Union[List, np.ndarray]) -> Tuple[int, int, int, int]:
    # Expect row = [class, score, x_min, y_min, x_max, y_max]
    x1 = int(round(row[2]))
    y1 = int(round(row[3]))
    x2 = int(round(row[4]))
    y2 = int(round(row[5]))
    return x1, y1, x2, y2


def crop_image_from_boxrow(image: Union[str, np.ndarray],
                           row: Union[List, np.ndarray],
                           det_results: Optional[List] = None) -> Union[np.ndarray, Tuple[np.ndarray, List]]:
    """Crop a single image region specified by a detection row.
    
    支持坐标映射：可将原始图像中的 OCR 检测坐标映射到裁剪后图像的坐标系统。

    Args:
        image: path to image or image array (BGR as read by cv2).
        row: array-like row in the format [class, score, x_min, y_min, x_max, y_max].
        det_results: (可选) OCR 检测结果列表，格式为 [[[pts], (text, score)], ...]
                    如果提供，会返回映射后的坐标。

    Returns:
        如果不提供 det_results：
            Cropped image (ndarray) corresponding to the box.
        
        如果提供 det_results：
            (cropped_image, mapped_det_results)
            其中 mapped_det_results 是坐标映射后的检测结果列表
    """
    # load image if path provided
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
    else:
        img = image

    if img is None:
        raise ValueError("image is None")

    h, w = img.shape[:2]
    # convert row to numpy array for indexing safety
    row_arr = np.asarray(row)
    if row_arr.size < 6:
        raise ValueError("row must have at least 6 elements: [class, score, x1, y1, x2, y2]")

    x1, y1, x2, y2 = _box_row_to_xyxy(row_arr)

    # clamp to image bounds
    x1 = _clamp(x1, 0, w - 1)
    x2 = _clamp(x2, 0, w - 1)
    y1 = _clamp(y1, 0, h - 1)
    y2 = _clamp(y2, 0, h - 1)

    if x2 < x1 or y2 < y1:
        raise ValueError(f"Invalid box after clamping: {(x1,y1,x2,y2)}")

    cropped_img = img[y1:y2 + 1, x1:x2 + 1].copy()
    
    # 如果提供了检测结果，进行坐标映射
    if det_results is not None:
        mapped_results = _map_det_coords_to_cropped(det_results, x1, y1, x2, y2)
        return cropped_img, mapped_results
    cv2.imwrite('./temp.jpg',cropped_img)
    print('\n saved to ./temp.jpg \n')
    return cropped_img


def _map_det_coords_to_cropped(det_results: List, 
                               crop_x1: int, crop_y1: int, 
                               crop_x2: int, crop_y2: int) -> List:
    """
    将原始图像中的 OCR 检测坐标映射到裁剪后的图像坐标系统。
    
    Args:
        det_results: OCR 检测结果列表，格式为 [[[pts], (text, score)], ...]
                    其中 pts 是四点坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        crop_x1, crop_y1: 裁剪区域左上角坐标（原始图像坐标系）
        crop_x2, crop_y2: 裁剪区域右下角坐标（原始图像坐标系）
    
    Returns:
        映射后的检测结果列表，坐标转换为裁剪后图像的坐标系统
    """
    mapped_results = []
    
    for det in det_results:
        if not det or len(det) < 2:
            continue
            
        ocr_poly = np.array(det[0])
        text, score = det[1]
        
        # 检查多边形是否在裁剪区域内
        # 只保留与裁剪区域有交集的检测结果
        poly_x_min = np.min(ocr_poly[:, 0])
        poly_y_min = np.min(ocr_poly[:, 1])
        poly_x_max = np.max(ocr_poly[:, 0])
        poly_y_max = np.max(ocr_poly[:, 1])
        
        # 检查是否与裁剪区域相交
        if (poly_x_max < crop_x1 or poly_x_min > crop_x2 or
            poly_y_max < crop_y1 or poly_y_min > crop_y2):
            # 不相交，跳过
            continue
        
        # 坐标映射：原始坐标 - 裁剪区域的起点
        mapped_poly = ocr_poly.copy()
        mapped_poly[:, 0] -= crop_x1  # 减去左边界
        mapped_poly[:, 1] -= crop_y1  # 减去上边界
        
        # 坐标裁剪：确保映射后的坐标在裁剪后的图像范围内
        crop_h = crop_y2 - crop_y1 + 1
        crop_w = crop_x2 - crop_x1 + 1
        
        mapped_poly[:, 0] = np.clip(mapped_poly[:, 0], 0, crop_w - 1)
        mapped_poly[:, 1] = np.clip(mapped_poly[:, 1], 0, crop_h - 1)
        
        # 构建映射后的检测结果
        mapped_det = [mapped_poly.tolist(), (text, score)]
        mapped_results.append(mapped_det)
    
    return mapped_results



def crop_tables_from_boxres(image: Union[str, np.ndarray],
                            boxres: Dict,
                            labels: List[str],
                            target_labels: Tuple[str, str] = ('table2', 'table'),
                            output_dir: Optional[str] = None,
                            prefix: str = 'crop') -> List[Dict]:
    """Crop boxes whose class name is in target_labels.

    Args:
        image: path to image or an image array (BGR as read by cv2).
        boxres: detection result dict with keys 'boxes' and 'boxes_num'.
        labels: list mapping class ids to names.
        target_labels: tuple/list of label names to crop (default ('table2','table')).
        output_dir: if provided, crops will be saved there.
        prefix: filename prefix when saving crops.

    Returns:
        List of dicts containing: 'label', 'score', 'box' (x1,y1,x2,y2), 'crop' (ndarray), 'save_path' (or None)
    """
    # Load image
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image}")
        image_path = image
    else:
        img = image
        image_path = None

    h, w = img.shape[:2]

    # Prepare output dir
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    boxes = boxres.get('boxes')
    # boxes could be concatenated np.ndarray of shape [N,6]
    if boxes is None:
        return []

    # Some detectors return boxes as np.ndarray with shape (N,6)
    # where first column is class id (float), second is score.
    results = []

    # Ensure numpy array
    boxes = np.asarray(boxes)

    # If boxes is empty
    if boxes.size == 0:
        return results

    for i in range(boxes.shape[0]):
        row = boxes[i]
        class_id = int(row[0])
        score = float(row[1])
        # Guard: class_id may be out of range
        if class_id < 0 or class_id >= len(labels):
            class_name = f"cls_{class_id}"
        else:
            class_name = labels[class_id]

        if class_name in target_labels:
            x1, y1, x2, y2 = _box_row_to_xyxy(row)
            # clamp
            x1 = _clamp(x1, 0, w - 1)
            x2 = _clamp(x2, 0, w - 1)
            y1 = _clamp(y1, 0, h - 1)
            y2 = _clamp(y2, 0, h - 1)
            # ensure x1<=x2 and y1<=y2
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            crop = img[y1:y2 + 1, x1:x2 + 1].copy()
            save_path = None
            if output_dir and image_path:
                base = os.path.splitext(os.path.basename(image_path))[0]
                out_name = f"{prefix}_{base}_{i}_{class_name}.jpg"
                save_path = os.path.join(output_dir, out_name)
                cv2.imwrite(save_path, crop)

            results.append({
                'label': class_name,
                'score': score,
                'box': [x1, y1, x2, y2],
                'crop': crop,
                'save_path': save_path
            })

    return results


def load_layout_from_file(layout_path: str) -> Tuple[Dict, List[str]]:
    """Load layout (boxres, labels) from a file.

    Supports:
      - JSON file containing an object with keys 'boxres' and 'labels' or a 2-item list [boxres, labels]
      - NPZ/NPY files (expects arrays saved under 'boxes' and 'boxes_num' or a dict/tuple)
      - Python file: executed via runpy.run_path() and expected to define `layoutres` or `boxres`+`labels`.

    WARNING: executing arbitrary Python files runs code. Only load trusted files.
    """
    p = Path(layout_path)
    if not p.exists():
        raise FileNotFoundError(layout_path)

    # JSON
    if p.suffix.lower() in {'.json', '.txt'}:
        # try parse as JSON first
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            # data can be { 'boxres': {...}, 'labels': [...] } or [boxres, labels]
            if isinstance(data, dict) and 'boxres' in data and 'labels' in data:
                return data['boxres'], data['labels']
            if isinstance(data, (list, tuple)) and len(data) == 2:
                return data[0], data[1]
        except Exception:
            # fall through to try executing as python-like text file
            pass

    # NPZ / NPY
    if p.suffix.lower() in {'.npz', '.npy'}:
        try:
            loaded = np.load(str(p), allow_pickle=True)
            # if loaded is an archive
            if isinstance(loaded, np.lib.npyio.NpzFile):
                # try common keys
                if 'boxres' in loaded and 'labels' in loaded:
                    return loaded['boxres'].item(), list(loaded['labels'])
                # attempt to build boxres from arrays
                if 'boxes' in loaded and 'boxes_num' in loaded:
                    boxres = {'boxes': loaded['boxes'], 'boxes_num': loaded['boxes_num']}
                    labels = list(loaded['labels']) if 'labels' in loaded else []
                    return boxres, labels
            else:
                # single array
                data = loaded
                return data.item(), []
        except Exception:
            pass

    # As a last resort, try executing the file as Python and inspect variables.
    # Warning: this runs arbitrary code. Only use trusted files.
    try:
        ns = runpy.run_path(str(p))
    except Exception as e:
        raise RuntimeError(f"Failed to load layout file {layout_path}: {e}")

    # look for layoutres, or boxres+labels
    if 'layoutres' in ns:
        layoutres = ns['layoutres']
        if isinstance(layoutres, (list, tuple)) and len(layoutres) == 2:
            return layoutres[0], layoutres[1]

    if 'boxres' in ns and 'labels' in ns:
        return ns['boxres'], ns['labels']

    # try other likely names
    for name in ('derres', 'layout', 'layout_res'):
        if name in ns:
            val = ns[name]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                return val[0], val[1]

    raise RuntimeError(f"Cannot find (boxres, labels) in {layout_path}")


def crop_tables_from_layout_file(layout_path: str,
                                 image: Union[str, np.ndarray],
                                 target_labels: Tuple[str, str] = ('table2', 'table'),
                                 output_dir: Optional[str] = None) -> List[Dict]:
    """Load layout from file and crop target table labels from image.

    Returns same structure as crop_tables_from_boxres.
    """
    boxres, labels = load_layout_from_file(layout_path)
    return crop_tables_from_boxres(image, boxres, labels, target_labels=target_labels, output_dir=output_dir)


def crop_tables_from_label_file(label_txt: str,
                                image_dir: str,
                                target_labels: Tuple[str, str] = ('table2', 'table'),
                                output_dir: Optional[str] = None,
                                prefix: str = 'crop') -> List[Dict]:
    """Parse a Label.txt file (lines: <image_path>\t<json_annotations>) and crop target labels.

    The Label.txt format in this repo is a TSV where the first column is the image path
    (often a relative path like 'TaxDoc/main_1_page1.jpg') and the second column is a JSON
    array of annotation objects with fields including 'points' and 'key_cls'.

    Args:
        label_txt: path to Label.txt
        image_dir: directory that contains the images referenced in Label.txt. The code will
                   try joining image_dir with the relative path from Label.txt. If that fails
                   it will try to match by basename (case-insensitive) with a recursive search.
        target_labels: labels to crop (default ('table2','table')).
        output_dir: if provided, save crops there.
        prefix: filename prefix when saving crops.

    Returns:
        List of crop metadata dicts similar to crop_tables_from_boxres.
    """
    p = Path(label_txt)
    if not p.exists():
        raise FileNotFoundError(label_txt)

    image_root = Path(image_dir)
    if not image_root.exists():
        raise FileNotFoundError(f"image_dir not found: {image_dir}")

    results = []
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Read lines
    with p.open('r', encoding='utf-8') as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue

            # Expect tab separated: rel_path \t json
            parts = line.split('\t', 1)
            if len(parts) == 1:
                # try whitespace split
                parts = line.split(None, 1)
                if len(parts) == 1:
                    # malformed line
                    continue

            rel_path = parts[0].strip()
            ann_json = parts[1].strip()

            # Resolve image path candidates
            candidates = []
            # If rel_path is absolute, try as-is
            rp = Path(rel_path)
            if rp.is_absolute():
                candidates.append(rp)
            # Join with provided image_dir
            candidates.append(image_root / rp)
            # try only basename under image_root (recursive)
            candidates.append(image_root / rp.name)

            image_path = None
            for cand in candidates:
                if cand.exists():
                    image_path = str(cand)
                    break

            # fallback: recursive search by name (case-insensitive)
            if image_path is None:
                name_lower = rp.name.lower()
                found = None
                for f in image_root.rglob('*'):
                    if f.is_file() and f.name.lower() == name_lower:
                        found = f
                        break
                if found:
                    image_path = str(found)

            if image_path is None:
                # couldn't find image; skip but record debug info
                # (do not raise so one missing file won't stop the whole batch)
                print(f"[warn] line {lineno}: image not found for '{rel_path}' under '{image_dir}'")
                continue

            # parse annotations json
            try:
                anns = json.loads(ann_json)
            except Exception as e:
                print(f"[warn] line {lineno}: failed to parse json: {e}")
                continue

            # load image once
            img = cv2.imread(image_path)
            if img is None:
                print(f"[warn] line {lineno}: cannot read image {image_path}")
                continue
            h, w = img.shape[:2]

            # iterate annotation entries
            for idx, ann in enumerate(anns):
                # ann expected: {'transcription':..., 'points': [[x,y],...], 'key_cls': 'table2'}
                key_cls = ann.get('key_cls') or ann.get('label') or ann.get('cls')
                if key_cls not in target_labels:
                    continue
                pts = ann.get('points')
                if not pts:
                    continue
                # polygon -> bbox
                xs = [int(round(p[0])) for p in pts]
                ys = [int(round(p[1])) for p in pts]
                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(w - 1, max(xs))
                y2 = min(h - 1, max(ys))
                if x2 < x1 or y2 < y1:
                    continue
                crop = img[y1:y2 + 1, x1:x2 + 1].copy()
                save_path = None
                if output_dir:
                    base = os.path.splitext(os.path.basename(image_path))[0]
                    out_name = f"{prefix}_{base}_{idx}_{key_cls}.jpg"
                    save_path = str(Path(output_dir) / out_name)
                    cv2.imwrite(save_path, crop)

                results.append({
                    'image': image_path,
                    'label': key_cls,
                    'box': [x1, y1, x2, y2],
                    'crop': crop,
                    'save_path': save_path,
                    'line': lineno,
                })

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Crop table/table2 boxes from image using layout file')
    parser.add_argument('--layout', required=True, help='Path to layout file (json/npz/py/txt)')
    parser.add_argument('--image', help='Path to source image (for single-image layouts)')
    parser.add_argument('--image-dir', help='Directory that contains images referenced by a multi-line Label.txt')
    parser.add_argument('--out', default='crops', help='Output directory for crops')
    args = parser.parse_args()
    out = args.out
    layout_path = args.layout

    # If an image-dir was provided, assume layout is a multi-line Label.txt and process all lines
    if args.image_dir:
        print(f"Processing Label file {layout_path} and locating images under {args.image_dir}...")
        crops = crop_tables_from_label_file(layout_path, args.image_dir, output_dir=out)
        print(f"Saved {len([c for c in crops if c.get('save_path')])} crops to {out} (total {len(crops)} items)")
    else:
        if not args.image:
            raise SystemExit("Either --image or --image-dir must be provided")
        print(f"Loading layout from {layout_path}... (will execute if python file)\n")
        crops = crop_tables_from_layout_file(layout_path, args.image, output_dir=out)
        print(f"Saved {len([c for c in crops if c.get('save_path')])} crops to {out} (total {len(crops)} items)")
