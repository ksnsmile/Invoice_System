# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:08:28 2025

@author: user
"""

import easyocr
import cv2
import os
import time

class KoreanTextFilter:
    def __init__(self):
        # EasyOCR reader 객체 생성 (한국어 지원)
        self.reader = easyocr.Reader(['ko'])

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
        for filename in os.listdir(warped_directory):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(warped_directory, filename)
                image = cv2.imread(image_path)

                # 한국어 텍스트가 포함된 경우 필터링
                if self.contains_korean_text(image):
                    filtered_save_path = os.path.join(filtered_directory, filename)
                    cv2.imwrite(filtered_save_path, image)
                    print(f"Filtered and saved: {filtered_save_path}")


# 메인 실행 부분
if __name__ == "__main__":
    # 디렉토리 경로 설정
    warped_directory = 'crops/warped_invoice_5'
    filtered_directory = 'crops/warped_invoice_filter_5'

    # KoreanTextFilter 객체 생성
    korean_filter = KoreanTextFilter()

    # 수행 시간 측정 시작
    start_time = time.time()

    # 이미지 필터링 실행
    korean_filter.filter_images_in_directory(warped_directory, filtered_directory)

    # 수행 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total elapsed time for filtering: {elapsed_time:.2f} seconds")