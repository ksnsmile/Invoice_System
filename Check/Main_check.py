# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:38:50 2025

@author: user
"""
import os
import time
import torch
from PIL import Image
import easyocr
import matplotlib.pyplot as plt
import numpy as np
from transformers import BertTokenizerFast
from torchvision.transforms import ToTensor
from Check.LayoutLM import LayoutLMForTokenClassification
from Check.Evaluate import *
from Check.Preprocessing import is_image_file



def Main_check():
    try:
        print("초기화 시작...")
        # 경로 설정
        
        folder_path = r"C:\Users\user\Desktop\ksn\invocie_system_2\Preprocessing\rotated_and_filtered_invoice_5"
        font_path = r"C:\Users\user\Desktop\ksn\invocie_system_2\Font\NotoSansKR-Medium.ttf"
        
        print("모델 로딩...")
        # EasyOCR 초기화
        reader = easyocr.Reader(['ko', 'en'], model_storage_directory=r'C:\Users\user\Desktop\ksn\invocie_system_2\EasyOCR\workspace\user_network_dir', 
                            user_network_directory=r'C:\Users\user\Desktop\ksn\invocie_system_2\EasyOCR\workspace\user_network_dir', 
                            recog_network='custom')
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = LayoutLMForTokenClassification()
        model.load_state_dict(torch.load(r"C:\Users\user\Desktop\ksn\invocie_system_2\trained_model.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        # 결과 저장을 위한 변수들
        overall_results = []
        timer = Timer()

        # 데이터 미리 로드
        print("데이터 로드 시작...")
        preprocessed_data = {}
        for file_name in os.listdir(folder_path):
            if is_image_file(file_name):
                print(f"Loading {file_name}...")
                file_path = os.path.join(folder_path, file_name)
                
                
                try:
                    image = Image.open(file_path).convert("RGB")
    
                    preprocessed_data[file_name] = {
                        'image': image,
                        'file_path': file_path,
                        
                    }
                    print(f"Successfully loaded {file_name}")
                except Exception as e:
                    print(f"Error loading {file_name}: {str(e)}")
                    continue

        print(f"총 {len(preprocessed_data)}개의 이미지 로드 완료")

        # 각 이미지 처리
        for file_name, data in preprocessed_data.items():
            print(f"\nProcessing image: {file_name}")
            print("-" * 50)
            
            try:
                # 이미지 입력
                print("이미지 처리 중...")
                start_time = time.time()
                image = data['image']
                timer.add_time('image_input', time.time() - start_time)

                # OCR 처리
                print("OCR 처리 중...")
                start_time = time.time()
                easyocr_results = reader.readtext(data['file_path'])
                print(f"OCR 결과: {len(easyocr_results)} items found")
                timer.add_time('ocr_process', time.time() - start_time)

                # LayoutLM 처리
                print("\nLayoutLM 처리 중...")
                start_time = time.time()
                try:
                    # EasyOCR 결과를 LayoutLM 처리에 전달
                    layoutlm_inputs = process_image_for_layoutlm(
                        data['file_path'], 
                        tokenizer, 
                        device,
                        easyocr_results  # OCR 결과 전달
                    )

                    print("\nLayoutLM 입력 형태:")
                    for key, value in layoutlm_inputs.items():
                        if isinstance(value, torch.Tensor):
                            print(f"{key}: shape {value.shape}, dtype {value.dtype}")

                    # LayoutLM 추론
                    model.eval()
                    with torch.no_grad():
                        layoutlm_outputs = model(
                            input_ids=layoutlm_inputs['input_ids'],
                            bbox=layoutlm_inputs['bbox'],
                            attention_mask=layoutlm_inputs['attention_mask'],
                            token_type_ids=layoutlm_inputs['token_type_ids'],
                            resized_images=layoutlm_inputs['resized_image'],
                            resized_and_aligned_bounding_boxes=layoutlm_inputs['resized_and_aligned_bounding_boxes']
                        )
                        
                        layoutlm_results = process_layoutlm_outputs(
                            layoutlm_outputs,
                            tokenizer,
                            layoutlm_inputs['bbox'],
                            image,
                            easyocr_results  
                        )
                        print(f"\nLayoutLM 결과: {len(layoutlm_results)} items found")
                    
                    timer.add_time('layoutlm_process', time.time() - start_time)

                    # 결과 처리
                    print("결과 처리 중...")
                    start_time = time.time()

                   
                    
                    # 결과 시각화
                    img = visualize_comparison(image, easyocr_results, layoutlm_results, font_path)
                    timer.add_time('postprocess', time.time() - start_time)

                except Exception as e:
                    print(f"LayoutLM 처리 중 오류 발생: {str(e)}")
                    layoutlm_results = []
                    results = {
                        'label_metrics': {},
                        'macro_f1': 0.0,
                        'weighted_f1': 0.0
                    }
                    

            
                # 처리 시간 출력
                current_times = timer.get_current_times()
                print("\nProcessing times:")
                for stage, t in current_times.items():
                    print(f"{stage}: {t:.4f} seconds")

                # 시각화 결과 표시
                plt.figure(figsize=(15, 10))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"LayoutLM Detection Results : {file_name}")
                plt.savefig(fr"C:\Users\user\Desktop\ksn\invocie_system_2\Results\{file_name}.png")
                plt.show()
                plt.close()

            except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    continue

        
    except Exception as e:
        print(f"Main function error: {str(e)}")

    finally:
        print("처리 완료")


        # 평균 처리 시간 출력
        avg_times = timer.get_average_times()
        print("\nAverage Processing Times:")
        for stage, avg_time in avg_times.items():
            print(f"{stage}: {avg_time:.4f} seconds")

if __name__ == "__main__":
    Main_check()