import cv2
import matplotlib.pyplot as plt
import sys,os
import copy 
import numpy as np
sys.path.append( '/data/tensorflow/kath/OCR/PaddleOCR-release-2.7/ppstructure') 
from datetime import datetime


def img_show(img):
    plt.figure('res', figsize=(12, 23))
    plt.imshow(img)
    plt.show()
def redink_remover(img):

    image = copy.deepcopy(img)
    blue_c, green_c, red_c = cv2.split(image)
    result_img = np.expand_dims( red_c, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

    return result_img


def point_trans_dt(w,h,dt_boxes,expand,width_ex):
    r_boxes = []
    # dt_boxes = sorted_boxes(np.array(dt_boxes))
    for box in dt_boxes:
        x_min = max(0, box[:, 0].min() - width_ex)
        x_max = min(w, box[:, 0].max() + width_ex)
        y_min = max(0, box[:, 1].min() - expand)
        y_max = min(h, box[:, 1].max() + expand)
        box = [x_min, y_min, x_max, y_max]
        r_boxes.append(box)
    return np.array(r_boxes)
 
def calculate_iou(rect1, rect2):
    """
    ¼ÆËãÁ½¸ö¾ØÐÎµÄIOU£¨Intersection over Union£©

    :param rect1: ¾ØÐÎ1µÄ×ø±ê [x1, y1, x2, y2]£¬ÆäÖÐ (x1, y1) Îª×óÉÏ½Ç£¬(x2, y2) ÎªÓÒÏÂ½Ç
    :param rect2: ¾ØÐÎ2µÄ×ø±ê [x1, y1, x2, y2]
    :return: IOUÖµ
    """
    # ¾ØÐÎ1
    x1_1, y1_1, x2_1, y2_1 = rect1
    # ¾ØÐÎ2
    x1_2, y1_2, x2_2, y2_2 = rect2

    # ¼ÆËãÏà½»¾ØÐÎµÄ×ø±ê
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    # ¼ÆËã½»¼¯¾ØÐÎµÄ¿íºÍ¸ß
    inter_width = max(0, inter_x2 - inter_x1)
    inter_height = max(0, inter_y2 - inter_y1)

    # ¼ÆËã½»¼¯Ãæ»ý
    inter_area = inter_width * inter_height

    # ¼ÆËãÁ½¸ö¾ØÐÎµÄÃæ»ý
    rect1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    rect2_area = (x2_2 - x1_2) * (y2_2 - y1_2)

    # ¼ÆËã²¢¼¯Ãæ»ý
    union_area = rect1_area + rect2_area - inter_area

    # ¼ÆËãIOU
    iou = inter_area / rect1_area if rect1_area != 0 else 0

    return iou
def point_trans_dt(w,h,dt_boxes,expand,width_ex):
    r_boxes = []
    # dt_boxes = sorted_boxes(np.array(dt_boxes))
    for box in dt_boxes:
        box=np.array(box)
        x_min = max(0, box[:, 0].min() - width_ex)
        x_max = min(w, box[:, 0].max() + width_ex)
        y_min = max(0, box[:, 1].min() - expand)
        y_max = min(h, box[:, 1].max() + expand)
        box = [x_min, y_min, x_max, y_max]
        r_boxes.append(box)
    return np.array(r_boxes)


def structure_box_trans(structure,ww,hh,w,h):
    _,boxes=structure
    firstbox=1
    firstbox_=[]
    for i in boxes:
        if firstbox==0:
            firstbox_=i
        i[0]=max(0,i[0]-ww)
        i[1]=max(0,i[1]-hh)

        i[2]=min(w,i[2]+ww)
        i[3]=max(0,i[3]-hh)

        i[4]=min(w,i[4]+ww)
        i[5]=min(h,i[5]+hh)

        i[6]=max(0,i[6]-ww)
        i[7]=min(h,i[7]+hh)
    return _,boxes
def visualize_table_annotation(img, bboxes, html_tokens=None, score=None):

    

    for i, box in enumerate(bboxes):
        pts = box.reshape(4, 2).astype(int)   # 转为4个点
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

        # 在左上角写上索引
        x, y = pts[0]
        cv2.putText(img, f"{i+1}", (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 如果有score，也显示一下
    if score is not None:
        cv2.putText(img, f"Score: {score:.3f}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 显示 — 保存图像并将生成时间加入文件名，方便排查与历史记录
    resimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_name = f"./tableresimg_{timestamp}.jpg"
    cv2.imwrite(out_name, resimg)

# def TabelMatch_model(h,w):
#     ppocr_root = '/data/tensorflow/kath/OCR/PaddleOCR-release-2.7'
#     if ppocr_root not in sys.path:
#         sys.path.insert(0, ppocr_root)
    
#     # Import with full module path
#     from ppstructure.table.matcher import TableMatch, distance, compute_iou
#     # from table.matcher import TableMatch,distance,compute_iou
    
#     def new_match_result(self, dt_boxes, pred_bboxes):
#         # print('---------------------------------------------------------------------')
#         # print('h,w=',h,w)
#         pred_box_weight={i:0 for i in range(len(pred_bboxes)) }
#         matched = {}
#         max_dis=((w-1)+(h-1))/6
#         for i, gt_box in enumerate(dt_boxes):
#             distances = []
#             tool_distance={}
#             for j, pred_box in enumerate(pred_bboxes):

#                 if len(pred_box) == 8:
#                     pred_box = [
#                         np.min(pred_box[0::2]), np.min(pred_box[1::2]),
#                         np.max(pred_box[0::2]), np.max(pred_box[1::2])
#                     ]

#                 # distances.append((distance(gt_box, pred_box)/max_dis,
#                 #                   1. - compute_iou(gt_box, pred_box)
#                 #                   ))  # compute iou and l1 distance

#                 if (1. - calculate_iou(gt_box, pred_box))>0.99 and (distance(gt_box, pred_box)/max_dis)>0.3:
#                     distances.append((3,1. - calculate_iou(gt_box, pred_box)))
#                     tool_distance[j]=(3,1. - calculate_iou(gt_box, pred_box))
#                 else:
#                     distances.append(((distance(gt_box, pred_box)/max_dis)+(1. - calculate_iou(gt_box, pred_box)**2)+ pred_box_weight[j],
#                                       1. - calculate_iou(gt_box, pred_box)
#                                       ))  # compute iou and l1 distance
#                     tool_distance[j]=((distance(gt_box, pred_box)/max_dis)+(1. - calculate_iou(gt_box, pred_box)**2)+ pred_box_weight[j],
#                                       1. - calculate_iou(gt_box, pred_box)
#                                       )
                
#                 # distances.append((distance(gt_box, pred_box)/max_dis,
#                 #                   1. - calculate_iou(gt_box, pred_box)
#                 #                   ))  # compute iou and l1 distance
#             # print(i,gt_box)
#             # if i==140:
#             #     print('\n [imp]',i,tool_distance)
#             sorted_distances = distances.copy()
#             # select det box by iou and l1 distance
#             sorted_distances = sorted(
#                 sorted_distances, key=lambda item: (item[1]+item[0]))
#             # sorted_distances = sorted(
#             #     sorted_distances, key=lambda item: (item[0]))
#             # print('sorted_distances:',sorted_distances)
#             # print()
#             if sorted_distances[0][1]>0.99 and sorted_distances[0][0]>2  : #and sorted_distances[0][0]>1
#                 # print(distances.index(sorted_distances[0]), sorted_distances[0],rec_res[i],rec_res[distances.index(sorted_distances[0])])
#                 # print('i',i, 'continueÁË')
#                 continue
#             if distances.index(sorted_distances[0]) not in matched.keys():
#                 matched[distances.index(sorted_distances[0])] = [i]
#             else:
#                 matched[distances.index(sorted_distances[0])].append(i)
#             pred_box_weight[distances.index(sorted_distances[0])]+=0.6
            

        
#         # matched_key=sorted(matched)
#         # matched={key: matched[key] for key in matched_key}
#         # print('\n\npred_box_weight:',pred_box_weight)
#         # print('\nmatched:',matched)
#         # print('---------------------------------------------------------------------') 
#         return matched
    
    
#     TableMatch.match_result=new_match_result

#     match=TableMatch()#filter_ocr_result=True
#     return match

