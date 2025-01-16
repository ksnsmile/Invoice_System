# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:45:06 2025

@author: user
"""

# 테스트 데이터를 사용하여 학습된 모델을 평가
from typing import Dict, List
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from difflib import SequenceMatcher
import numpy as np
import os
import json
from PIL import ImageFont, ImageDraw, Image
import torch
from torchvision import transforms

class Timer:
    def __init__(self):
        self.times = {
            'image_input': [],
            'preprocess': [],
            'ocr_process': [],
            'postprocess': []
        }
    
    def add_time(self, stage: str, duration: float):
        if stage not in self.times:
            self.times[stage] = []
        self.times[stage].append(duration)
    
    def get_average_times(self) -> Dict[str, float]:
        return {stage: np.mean(times) if times else 0 
                for stage, times in self.times.items()}

    def get_current_times(self) -> Dict[str, float]:
        return {stage: times[-1] if times else 0 
                for stage, times in self.times.items()}



def convert_bbox_format(x, y, width, height, original_width, original_height):
    """실제 픽셀 좌표를 바운딩 박스 형식으로 변환"""
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + width)
    y2 = int(y + height)
    return [x1, y1, x2, y2]

def resize_bounding_box(bbox, original_image, target_size):
    """바운딩 박스를 타겟 크기에 맞게 리사이즈"""
    original_width, original_height = original_image.size
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    x_scale = new_width / original_width
    y_scale = new_height / original_height
    
    orig_left, orig_top, orig_right, orig_bottom = bbox
    return [
        int(np.round(orig_left * x_scale)),
        int(np.round(orig_top * y_scale)),
        int(np.round(orig_right * x_scale)),
        int(np.round(orig_bottom * y_scale))
    ]

def convert_points_to_bbox(points):
    """EasyOCR 포인트를 바운딩 박스 형식으로 변환"""
    x_coords = [int(p[0]) for p in points]
    y_coords = [int(p[1]) for p in points]
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

def calculate_iou(box1, box2):
    """두 바운딩 박스 간의 IoU 계산"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def visualize_comparison(image, easyocr_results, layoutlm_results, font_path):
    """EasyOCR 텍스트와 LayoutLM 레이블을 함께 시각화"""
    # 원본 이미지 크기 유지
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, 20)

    def calculate_iou(box1, box2):
        """두 바운딩 박스의 IoU 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0

    # 각 OCR 결과에 대해 가장 잘 매칭되는 LayoutLM 결과 찾기
    for result in easyocr_results:
        points = np.array(result[0])
        text = result[1]
        
        # OCR 바운딩 박스 계산
        ocr_bbox = [
            min(p[0] for p in points),
            min(p[1] for p in points),
            max(p[0] for p in points),
            max(p[1] for p in points)
        ]

        # 바운딩 박스는 빨간색으로 표시 (옵션)
        draw.rectangle(ocr_bbox, outline='red', width=2)
        
        # 가장 높은 IoU를 가진 LayoutLM 결과 찾기
        best_iou = 0
        matched_label = "unknown"
        
        for layout_result in layoutlm_results:
            iou = calculate_iou(ocr_bbox, layout_result['bbox'])
            if iou > best_iou:
                best_iou = iou
                matched_label = layout_result['label']

        # OCR 텍스트와 LayoutLM 레이블을 함께 표시
        # IoU가 충분히 높은 경우에만 레이블 표시 (임계값: 0.5)
        label_text = f"{text} ({matched_label})" if best_iou > 0.5 else text
        
        # 텍스트 위치 계산 (바운딩 박스 위에)
        text_x = ocr_bbox[0]
        text_y = ocr_bbox[1] - 25  # 바운딩 박스 위에 여백을 두고 표시
        
        # 배경색 추가로 가독성 향상
        text_bbox = draw.textbbox((text_x, text_y), label_text, font=font)
        draw.text((text_x, text_y), label_text, font=font, fill='green')

    return img

def is_image_file(filename):
    """이미지 파일인지 확인"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return os.path.splitext(filename.lower())[1] in image_extensions

def process_image_for_layoutlm(image_path, tokenizer, device, ocr_results):
    # 이미지 로드 및 전처리
    image = Image.open(image_path).convert("RGB")
    original_size = image.size
    
    # OCR 결과로부터 words와 bounding boxes 추출
    words = []
    bounding_boxes = []
    
    for detection in ocr_results:
        # EasyOCR 결과에서 텍스트와 바운딩 박스 추출
        text = detection[1]
        points = detection[0]
        
        # 빈 텍스트는 건너뛰기
        if not text.strip():
            
            continue
            
        words.append(text)
        
        # Points를 바운딩 박스로 변환
        x_coords = [int(p[0]) for p in points]
        y_coords = [int(p[1]) for p in points]
        bbox = [
            min(x_coords),  # x1
            min(y_coords),  # y1
            max(x_coords),  # x2
            max(y_coords)   # y2
        ]
        bounding_boxes.append(bbox)
    
    # 입력이 비어있는 경우 처리
    if not words:
        print(f"Warning: No valid OCR results found for {image_path}")
        # 더미 데이터 생성
        words = ["[PAD]"]
        bounding_boxes = [[0, 0, 0, 0]]
    
    # LayoutLM용 정규화된 바운딩 박스 생성
    normalized_boxes = []
    width, height = original_size
    for bbox in bounding_boxes:
        # 0으로 나누기 방지
        if width == 0 or height == 0:
            normalized_box = [0, 0, 0, 0]
        else:
            normalized_box = [
                min(max(int(1000 * bbox[0] / width), 0), 1000),
                min(max(int(1000 * bbox[1] / height), 0), 1000),
                min(max(int(1000 * bbox[2] / width), 0), 1000),
                min(max(int(1000 * bbox[3] / height), 0), 1000)
            ]
        normalized_boxes.append(normalized_box)
    
    # 토큰화 및 인코딩
    encoding = tokenizer(
        words,
        padding="max_length",
        truncation=True,
        max_length=512,
        is_split_into_words=True,
        return_tensors="pt"
    )
    
    # 바운딩 박스 처리
    token_boxes = []
    word_ids = encoding.word_ids()
    
    for word_id in word_ids:
        if word_id is None:
            token_boxes.append([0, 0, 0, 0])
        else:
            token_boxes.append(normalized_boxes[min(word_id, len(normalized_boxes)-1)])
    
    bbox_tensor = torch.tensor([token_boxes], dtype=torch.long).to(device)
    
    # 이미지 리사이즈 및 패딩
    max_size = 1024
    ratio = min(max_size / width, max_size / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    
    # 이미지 변환 및 정규화
    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    image_tensor = transform(image)
    
    # 패딩 추가
    padded_image = torch.zeros(3, max_size, max_size)
    padded_image[:, :new_height, :new_width] = image_tensor
    
    # device로 이동
    input_tensor = {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device),
        'token_type_ids': encoding['token_type_ids'].to(device),
        'bbox': bbox_tensor,
        'resized_image': padded_image.unsqueeze(0).to(device),
        'resized_and_aligned_bounding_boxes': bbox_tensor,
        'original_size': original_size,
        'words': words,
        'bounding_boxes': bounding_boxes
    }
    
    return input_tensor

def process_layoutlm_outputs(outputs, tokenizer, bboxes, image, ocr_results):
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=-1)
    
    # 원본 이미지 크기
    original_width, original_height = image.size
    
    results = []
    pred_labels = predictions[0].cpu().numpy()
    batch_bboxes = bboxes[0].cpu().numpy()
    
    # OCR 바운딩 박스 저장을 위한 집합
    processed_boxes = set()
    
    def calculate_iou(box1, box2):
        """두 바운딩 박스의 IoU(Intersection over Union) 계산"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
            
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def denormalize_bbox(bbox):
        """LayoutLM의 정규화된 좌표를 원본 이미지 크기로 변환"""
        return [
            int(bbox[0] * original_width / 1000),
            int(bbox[1] * original_height / 1000),
            int(bbox[2] * original_width / 1000),
            int(bbox[3] * original_height / 1000)
        ]
    
    def find_matching_ocr_box(denorm_bbox):
        """가장 잘 매칭되는 OCR 바운딩 박스 찾기"""
        best_iou = 0
        best_box = None
        
        for result in ocr_results:
            points = result[0]
            ocr_box = [
                min(p[0] for p in points),
                min(p[1] for p in points),
                max(p[0] for p in points),
                max(p[1] for p in points)
            ]
            
            iou = calculate_iou(denorm_bbox, ocr_box)
            if iou > best_iou:
                best_iou = iou
                best_box = ocr_box
        
        return best_box if best_iou > 0.5 else None
    
    current_bbox = None
    current_label = None
    current_entities = []
    
    for idx, (pred_id, bbox) in enumerate(zip(pred_labels, batch_bboxes)):
        if pred_id == -100 or not bbox.any():
            continue
            
        pred_label = idx2label.get(pred_id)
        if not pred_label:
            continue
        
        # 바운딩 박스를 원본 이미지 크기로 변환
        denorm_bbox = denormalize_bbox(bbox)
        
        # OCR 결과와 매칭
        matched_box = find_matching_ocr_box(denorm_bbox)
        if matched_box is None:
            continue
            
        # 중복 감지 방지
        box_key = tuple(matched_box)
        if box_key in processed_boxes:
            continue
        
        processed_boxes.add(box_key)
        
        # B- 태그로 시작하는 새로운 엔티티
        if pred_label.startswith('B-'):
            if current_bbox is not None and current_label is not None:
                current_entities.append({
                    'bbox': current_bbox,
                    'label': current_label[2:] if current_label.startswith('B-') or current_label.startswith('I-') else current_label,
                    'confidence': 1.0
                })
            
            current_bbox = matched_box
            current_label = pred_label
            
        # I- 태그이고 이전 엔티티와 일치하는 경우
        elif pred_label.startswith('I-') and current_label is not None:
            base_label = pred_label[2:]
            current_base_label = current_label[2:] if current_label.startswith('B-') or current_label.startswith('I-') else current_label
            
            if base_label == current_base_label:
                # 기존 바운딩 박스 유지
                current_bbox = [
                    min(current_bbox[0], matched_box[0]),
                    min(current_bbox[1], matched_box[1]),
                    max(current_bbox[2], matched_box[2]),
                    max(current_bbox[3], matched_box[3])
                ]
    
    # 마지막 엔티티 처리
    if current_bbox is not None and current_label is not None:
        current_entities.append({
            'bbox': current_bbox,
            'label': current_label[2:] if current_label.startswith('B-') or current_label.startswith('I-') else current_label,
            'confidence': 1.0
        })
    
    results.extend(current_entities)
    
    print(f"\nProcessed {len(results)} entities")
    print("Sample bounding boxes (first 3):")
    for i, result in enumerate(results[:3]):
        print(f"Entity {i+1}: {result['label']}")
        print(f"Bbox: {result['bbox']}")
    
    return results


labels = ['B-sign', 'I-sign',
'B-recipient_key', 'I-recipient_key',
'B-recipient_name', 'I-recipient_name',
'B-recipient_phone_number_key', 'I-recipient_phone_number_key',
'B-recipient_phone_number', 'I-recipient_phone_number',
'B-recipient_address_do', 'I-recipient_address_do',
'B-recipient_address_si', 'I-recipient_address_si',
'B-recipient_address_gun', 'I-recipient_address_gun',
'B-recipient_address_gu', 'I-recipient_address_gu',
'B-recipient_address_eup', 'I-recipient_address_eup',
'B-recipient_address_myeon', 'I-recipient_address_myeon',
'B-recipient_address_ri', 'I-recipient_address_ri',
'B-recipient_address_dong', 'I-recipient_address_dong',
'B-recipient_address_jibeon', 'I-recipient_address_jibeon',
'B-recipient_address_ro_name', 'I-recipient_address_ro_name',
'B-recipient_address_gil_name', 'I-recipient_address_gil_name',
'B-recipient_address_ro_number', 'I-recipient_address_ro_number',
'B-recipient_address_building_number', 'I-recipient_address_building_number',
'B-recipient_address_room_number', 'I-recipient_address_room_number',
'B-recipient_address_detail', 'I-recipient_address_detail',
'B-sender_key', 'I-sender_key',
'B-sender_name', 'I-sender_name',
'B-sender_phone_number_key', 'I-sender_phone_number_key',
'B-sender_phone_number', 'I-sender_phone_number',
'B-sender_address_do', 'I-sender_address_do',
'B-sender_address_si', 'I-sender_address_si',
'B-sender_address_gun', 'I-sender_address_gun',
'B-sender_address_gu', 'I-sender_address_gu',
'B-sender_address_eup', 'I-sender_address_eup',
'B-sender_address_myeon', 'I-sender_address_myeon',
'B-sender_address_ri', 'I-sender_address_ri',
'B-sender_address_dong', 'I-sender_address_dong',
'B-sender_address_jibeon', 'I-sender_address_jibeon',
'B-sender_address_ro_name', 'I-sender_address_ro_name',
'B-sender_address_gil_name', 'I-sender_address_gil_name',
'B-sender_address_ro_number', 'I-sender_address_ro_number',
'B-sender_address_building_number', 'I-sender_address_building_number',
'B-sender_address_room_number', 'I-sender_address_room_number',
'B-sender_address_detail', 'I-sender_address_detail',
'B-volume_key', 'I-volume_key',
'B-volume', 'I-volume',
'B-delivery_message_key', 'I-delivery_message_key',
'B-delivery_message', 'I-delivery_message',
'B-product_name_key', 'I-product_name_key',
'B-product_name', 'I-product_name',
'B-tracking_number_key', 'I-tracking_number_key',
'B-tracking_number', 'I-tracking_number',
'B-weight_key', 'I-weight_key',
'B-weight', 'I-weight',
'B-terminal_number', 'I-terminal_number',
'B-company_name', 'I-company_name',
'B-handwriting', 'I-handwriting',
'B-others', 'I-others'
]


idx2label = {v: k for v, k in enumerate(labels)}
label2idx = {k: v for v, k in enumerate(labels)}