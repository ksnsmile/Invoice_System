# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:09:42 2025

@author: user
"""

import cv2
import easyocr
import os
import time
import json

class ImageRotatorAndFilter:
    def __init__(self, common_words):
        # EasyOCR reader 객체 생성 (한국어 및 영어 지원)
        self.reader = easyocr.Reader(['ko', 'en'])
        # 자주 등장하는 단어 리스트
        self.common_words = common_words
        
        # 처리된 파일 목록을 파일에서 로드
        self.processed_files_path = r"C:\Users\user\Desktop\ksn\Invoice_System\processed_files.json"
        self.processed_files = self.load_processed_files()
        
    
    def load_processed_files(self):
        """처리된 파일 목록을 JSON 파일에서 로드"""
        if os.path.exists(self.processed_files_path):
            with open(self.processed_files_path, 'r') as f:
                return set(json.load(f))
        else:
            return set()

    def save_processed_files(self):
        """처리된 파일 목록을 JSON 파일에 저장"""
        with open(self.processed_files_path, 'w') as f:
            json.dump(list(self.processed_files), f)

    def rotate_image(self, image, angle):
        """이미지를 주어진 각도로 회전합니다."""
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)

        if angle == 90:
            rotated_image = cv2.transpose(image)
            rotated_image = cv2.flip(rotated_image, 1)
        elif angle == 180:
            rotated_image = cv2.flip(image, -1)
        elif angle == 270:
            rotated_image = cv2.transpose(image)
            rotated_image = cv2.flip(rotated_image, 0)

        return rotated_image

    def count_valid_words(self, text_list):
        """텍스트 리스트에서 자주 등장하는 단어의 개수를 세어 반환합니다."""
        valid_word_count = 0
        for text in text_list:
            for word, _ in self.common_words:
                if word in text:
                    valid_word_count += 1
        return valid_word_count

    def contains_common_words(self, text_list):
        """텍스트 리스트에 자주 등장하는 단어가 포함되어 있는지 확인합니다."""
        for text in text_list:
            for word, _ in self.common_words:
                if word in text:
                    return True
        return False

    def get_best_rotation(self, image):
        """이미지를 다양한 각도로 회전시켜 가장 많은 유효 단어를 포함하는 각도를 찾습니다."""
        angles = [0, 90, 180, 270]
        max_valid_words = 0
        best_angle = 0

        for angle in angles:
            rotated_image = self.rotate_image(image, angle)
            results = self.reader.readtext(rotated_image)
            texts = [text for (bbox, text, prob) in results]
            valid_word_count = self.count_valid_words(texts)

            if valid_word_count > max_valid_words:
                max_valid_words = valid_word_count
                best_angle = angle

        return best_angle
    
    
         

    def process_images_in_directory(self, image_dir, output_dir):
        """디렉토리 내의 모든 이미지를 처리하여 회전 및 필터링합니다."""
        os.makedirs(output_dir, exist_ok=True)
        

        for filename in os.listdir(image_dir) :
            if filename not in self.processed_files:
                # 처리된 파일 목록에 추가
                self.processed_files.add(filename)
                self.save_processed_files()
                image_path = os.path.join(image_dir, filename)
                
            
                image = cv2.imread(image_path)

                # 최적의 회전 각도 찾기
                best_rotation_angle = self.get_best_rotation(image)
                

                # 이미지 회전
                corrected_image = self.rotate_image(image, best_rotation_angle)

                # 회전된 이미지에서 텍스트 추출
                results = self.reader.readtext(corrected_image)
                texts = [text for (bbox, text, prob) in results]

                # 자주 등장하는 단어 포함 여부 확인
                if self.contains_common_words(texts):
                    
                    # 결과 이미지 저장
                    corrected_image_path = os.path.join(output_dir, f"corrected_{os.path.basename(image_path)}")
                    cv2.imwrite(corrected_image_path, corrected_image)
                    
                
                    return True
                else:
                    
                    
                    return False

            


# 메인 실행 부분
if __name__ == "__main__":
    # 이미지 디렉토리와 출력 디렉토리 설정
    image_dir = 'crops/warped_invoice_filter_5'  # 송장 이미지들이 저장된 디렉토리
    output_dir = 'crops/rotated_and_filtered_invoice_5'  # 결과 이미지를 저장할 디렉토리

    # 자주 등장하는 단어 리스트 설정 (예시)
    common_words = [
        ("운송장", 10), ("받는 분", 10), ("보내는 분", 10), ("받는", 10), ("보내는", 10), ("분", 10),
        ("서울", 5), ("인천", 5), ("경기", 5), ("강원", 5), ("충청", 5), ("전라", 5), ("경상", 5),
        ("시", 5), ("군", 5), ("구", 5), ("읍", 5), ("면", 5), ("동", 5), ("대", 5), ("비", 5),
        ("당", 5), ("대학", 3), ("국", 3), ("성동", 3), ("호", 3), ("특별", 3), ("별", 3)
    ]

    # ImageRotatorAndFilter 객체 생성
    image_processor = ImageRotatorAndFilter(common_words)

    # 이미지 처리 함수 호출
    image_processor.process_images_in_directory(image_dir, output_dir)