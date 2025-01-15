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
from LayoutLM import LayoutLMForTokenClassification
from Evaluate import *
from Preprocessing import is_image_file


#%%
def main():
    try:
        print("초기화 시작...")
        # 경로 설정
        annotation_dir = r"C:\Users\user\Desktop\system\content_combine\annotation_combine"
        folder_path = r"C:\Users\user\Desktop\system\content_combine\data_test2"
        font_path = r"C:\Users\user\Desktop\system\Font\NotoSansKR-Medium.ttf"
        
        print("모델 로딩...")
        # EasyOCR 초기화
        reader = easyocr.Reader(['ko', 'en'], model_storage_directory=r'C:\Users\user\Desktop\system\EasyOCR\workspace\user_network_dir', 
                            user_network_directory=r'C:\Users\user\Desktop\system\EasyOCR\workspace\user_network_dir', 
                            recog_network='custom')
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = LayoutLMForTokenClassification()
        model.load_state_dict(torch.load(r"C:\Users\user\Desktop\system\trained_model.pth"))
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
                annotation_path = os.path.join(annotation_dir, os.path.splitext(file_name)[0] + '.json')
                
                try:
                    image = Image.open(file_path).convert("RGB")
                    ground_truth = load_ground_truth(annotation_dir, file_name)
                    if ground_truth is None:
                        print(f"Warning: No ground truth found for {file_name}")
                        continue
                        
                    preprocessed_data[file_name] = {
                        'image': image,
                        'file_path': file_path,
                        'annotation_path': annotation_path,
                        'ground_truth': ground_truth
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
                        data['annotation_path'], 
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

                    results = evaluate_detection_and_text_performance(
                        easyocr_results,
                        layoutlm_results,
                        data['ground_truth'],
                        image  # image 인자 추가
                    )
                    
                    overall_results.append(results)
                    
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
                    overall_results.append(results)

                # 결과 출력
                print(f"\nResults for {file_name}:")
                print("\nLayoutLM Performance:")
                print(f"Macro F1-score: {results['macro_f1']:.4f}")
                print(f"Weighted F1-score: {results['weighted_f1']:.4f}")
                
                print("\nPer-label Performance:")
                for label, metrics in results['label_metrics'].items():
                    if metrics['support'] > 0:  # 해당 레이블이 데이터에 존재하는 경우만
                        print(f"\n{label}:")
                        print(f"  Precision: {metrics['precision']:.4f}")
                        print(f"  Recall: {metrics['recall']:.4f}")
                        print(f"  F1-score: {metrics['f1']:.4f}")
                        print(f"  Support: {metrics['support']}")

                # 처리 시간 출력
                current_times = timer.get_current_times()
                print("\nProcessing times:")
                for stage, t in current_times.items():
                    print(f"{stage}: {t:.4f} seconds")

                # 시각화 결과 표시
                plt.figure(figsize=(15, 10))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"LayoutLM Detection Results - Macro F1: {results['macro_f1']:.4f}")
                plt.show()
                plt.close()

            except Exception as e:
                    print(f"Error processing {file_name}: {str(e)}")
                    continue

        
    except Exception as e:
        print(f"Main function error: {str(e)}")

    finally:
        print("처리 완료")

    # 전체 평균 성능 출력
    if overall_results:
        print("\nOverall Average Performance:")
        
        # 모든 레이블에 대한 평균 메트릭 계산
        all_labels = set()
        for result in overall_results:
            all_labels.update(result['label_metrics'].keys())
        
        # 레이블별 평균 성능
        label_averages = {}
        for label in all_labels:
            metrics_for_label = []
            support_for_label = 0
            for result in overall_results:
                if label in result['label_metrics'] and result['label_metrics'][label]['support'] > 0:
                    metrics_for_label.append(result['label_metrics'][label])
                    support_for_label += result['label_metrics'][label]['support']
            
            if metrics_for_label:
                label_averages[label] = {
                    'precision': np.mean([m['precision'] for m in metrics_for_label]),
                    'recall': np.mean([m['recall'] for m in metrics_for_label]),
                    'f1': np.mean([m['f1'] for m in metrics_for_label]),
                    'support': support_for_label
                }
        
        # 전체 평균 F1-scores
        avg_macro_f1 = np.mean([r['macro_f1'] for r in overall_results])
        avg_weighted_f1 = np.mean([r['weighted_f1'] for r in overall_results])
        
        print(f"\nAverage Macro F1-score: {avg_macro_f1:.4f}")
        print(f"Average Weighted F1-score: {avg_weighted_f1:.4f}")

        # 평균 처리 시간 출력
        avg_times = timer.get_average_times()
        print("\nAverage Processing Times:")
        for stage, avg_time in avg_times.items():
            print(f"{stage}: {avg_time:.4f} seconds")

if __name__ == "__main__":
    main()