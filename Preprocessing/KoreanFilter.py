# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:08:28 2025

@author: user
"""

import easyocr
import cv2
import os
import time
import json

class KoreanTextFilter:
    def __init__(self):
        # EasyOCR reader 객체 생성 (한국어 지원)
        self.reader = easyocr.Reader(['ko'])

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
            
            
    def contains_korean_text(self, image):
        """이미지에서 한국어 텍스트가 포함되어 있는지 확인합니다."""
        results = self.reader.readtext(image)
        contains_korean = False
        only_numbers = True

        for (bbox, text, prob) in results:
            # 텍스트에 한글이 포함되어 있는지 확인
            if any('\uAC00' <= char <= '\uD7A3' for char in text):
                contains_korean = True
            # 텍스트가 숫자로만 구성되어 있는지 확인
            if not text.isdigit():
                only_numbers = False

        # 한국어가 포함되어 있고, 숫자로만 구성되지 않은 경우 True 반환
        return contains_korean and not only_numbers

    
         
    def filter_images_in_directory(self, warped_directory, filtered_directory):
        """디렉토리 내의 이미지에서 한국어 텍스트가 포함된 이미지를 필터링합니다."""
        # 필터링된 이미지를 저장할 디렉토리 생성
        if not os.path.exists(filtered_directory):
            os.makedirs(filtered_directory)
            print(f"Created directory: {filtered_directory}")

        # 디렉토리 내의 모든 JPG 파일 처리
        for filename in os.listdir(warped_directory) :
            if filename not in self.processed_files:
                image_path = os.path.join(warped_directory, filename)
                image = cv2.imread(image_path)

                # 한국어 텍스트가 포함된 경우 필터링
                if self.contains_korean_text(image):
                    # 처리된 파일 목록에 추가
                    self.processed_files.add(filename)
                    self.save_processed_files() 
                    base_filename, file_extension = os.path.splitext(filename)
                    save_filename = f"{base_filename}_filtered{file_extension}"
                    filtered_save_path = os.path.join(filtered_directory, save_filename)
                    cv2.imwrite(filtered_save_path, image)
                    print(f"Filtered and saved: {filtered_save_path}")
                     # 처리된 파일 목록 저장
                    return True
                else:
                    # 처리된 파일 목록에 추가
                    self.processed_files.add(filename)
                    self.save_processed_files()  # 처리된 파일 목록 저장
                    return False


# 메인 실행 부분
if __name__ == "__main__":
    # 디렉토리 경로 설정
    warped_directory = 'crops/warped_invoice_5'
    filtered_directory = 'crops/warped_invoice_filter_5'

    # KoreanTextFilter 객체 생성
    korean_filter = KoreanTextFilter()

   

    # 이미지 필터링 실행
    korean_filter.filter_images_in_directory(warped_directory, filtered_directory)

    