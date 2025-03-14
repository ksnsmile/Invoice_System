# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 14:24:49 2025

@author: user
"""

# main.py
import time
from Preprocessing.Wrap import DocumentProcessor
from Preprocessing.KoreanFilter import KoreanTextFilter
from Preprocessing.Rotate import ImageRotatorAndFilter

def Main_preprocessing():
    # 디렉토리 경로 설정
    input_directory = r'C:\Users\user\Desktop\ksn\Invoice_System\Preprocessing\crops\invoice'
    warped_directory = r'C:\Users\user\Desktop\ksn\Invoice_System\Preprocessing\warped_invoice_5'
    filtered_directory = r'C:\Users\user\Desktop\ksn\Invoice_System\Preprocessing\warped_invoice_filter_5'
    rotated_directory = r'C:\Users\user\Desktop\ksn\Invoice_System\Preprocessing\rotated_and_filtered_invoice_5'
    cut_directory = r'C:\Users\user\Desktop\ksn\Invoice_System\Preprocessing\cut_invoice'

    # 자주 등장하는 단어 리스트 설정 (예시)
    common_words = [
        ("운송장", 10), ("받는 분", 10), ("보내는 분", 10), ("받는", 10), ("보내는", 10), ("분", 10),
        ("서울", 5), ("인천", 5), ("경기", 5), ("강원", 5), ("충청", 5), ("전라", 5), ("경상", 5),
        ("시", 5), ("군", 5), ("구", 5), ("읍", 5), ("면", 5), ("동", 5), ("대", 5), ("비", 5),
        ("당", 5), ("대학", 3), ("국", 3), ("성동", 3), ("호", 3), ("특별", 3), ("별", 3)
    ]

    

     # 1. wrap.py 실행: 이미지 왜곡 보정
    processor = DocumentProcessor()
    flag = processor.process_images_in_directory(input_directory, warped_directory, cut_directory)

    if flag:
        # 2. character_filtered.py 실행: 한국어 텍스트 필터링
        korean_filter = KoreanTextFilter()
        flag_2 = korean_filter.filter_images_in_directory(warped_directory, filtered_directory)

        if flag_2:
            # 3. rotate.py 실행: 이미지 회전 및 필터링
            image_processor = ImageRotatorAndFilter(common_words)
            flag_3 = image_processor.process_images_in_directory(filtered_directory, rotated_directory)

            if flag_3 == False:
                print('회전 실패')
                return False
        else:
            print('필터링 실패')
            return False
    else:
        print('역원근 실패')
        return False    
    
    
    return flag_3

if __name__ == "__main__":
    Main_preprocessing()