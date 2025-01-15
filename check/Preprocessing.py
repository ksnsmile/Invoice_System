# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:41:59 2025

@author: user
"""
from PIL import Image
import numpy as np
import os
import json

def resize_bounding_box(bbox, original_image, target_size):
    # 원본 이미지의 가로, 세로 크기
    original_width, original_height = original_image.size

    # 이미지의 가로/세로 비율을 계산
    aspect_ratio = original_width / original_height

    # 타겟 크기의 비율에 맞게 조정
    if aspect_ratio > 1:
        # 가로가 더 길면, 타겟 크기의 가로를 기준으로 세로 크기 조정
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        # 세로가 더 길면, 타겟 크기의 세로를 기준으로 가로 크기 조정
        new_height = target_size
        new_width = int(target_size * aspect_ratio)

    # 스케일 팩터 계산
    x_scale = new_width / original_width
    y_scale = new_height / original_height
    
    # 바운딩 박스 좌표에 비율을 적용하여 크기 조정
    orig_left, orig_top, orig_right, orig_bottom = bbox

    # 좌표를 새롭게 조정
    resized_left = int(np.round(orig_left * x_scale))
    resized_top = int(np.round(orig_top * y_scale))
    resized_right = int(np.round(orig_right * x_scale))
    resized_bottom = int(np.round(orig_bottom * y_scale))

    # 새로운 좌표 반환
    return [resized_left, resized_top, resized_right, resized_bottom]

def align_bounding_boxes(bounding_boxes):
    aligned_bounding_boxes = []
    for bbox in bounding_boxes:
        # 약간의 오프셋을 주어 좌표를 조정합니다.
        aligned_bbox = [bbox[0] - 0.5, bbox[1] - 0.5, bbox[2] + 0.5, bbox[3] + 0.5]
        aligned_bounding_boxes.append(aligned_bbox)
    return aligned_bounding_boxes

def normalize_box(box, width, height):
     return [
         int((box[0] / width)),
         int((box[1] / height)),
         int((box[2] / width)),
         int((box[3] / height)),
     ]

def resize_and_align_bounding_box(bbox, original_image, target_size):
  x_, y_ = original_image.size

  x_scale = target_size / x_
  y_scale = target_size / y_

  origLeft, origTop, origRight, origBottom = tuple(bbox)

  x = int(np.round(origLeft * x_scale))
  y = int(np.round(origTop * y_scale))
  xmax = int(np.round(origRight * x_scale))
  ymax = int(np.round(origBottom * y_scale))

  return [x-0.5, y-0.5, xmax+0.5, ymax+0.5]

def is_image_file(filename):
    """이미지 파일인지 확인"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp'}
    return os.path.splitext(filename.lower())[1] in image_extensions
