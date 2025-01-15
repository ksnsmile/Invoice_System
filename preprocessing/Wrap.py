# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:39:38 2025

@author: user
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 13:39:38 2025

@author: user
"""

import cv2
import numpy as np
import os
import time

class DocumentProcessor:
    def __init__(self):
        pass

    def order_points(self, pts):
        """주어진 4개의 점을 정렬합니다 (상단 왼쪽, 상단 오른쪽, 하단 오른쪽, 하단 왼쪽)."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def find_document_edges(self, image):
        """이미지에서 문서의 가장자리를 찾습니다."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        image_area = image.shape[0] * image.shape[1]

        for contour in contours:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) == 4:
                rect = cv2.boundingRect(approx)
                contour_area = rect[2] * rect[3]
                if contour_area > 0.1 * image_area:
                    return approx

        c = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        width = rect[1][0]
        height = rect[1][1]
        box_area = width * height

        if box_area < 0.2 * image_area or box_area > 0.8 * image_area:
            return None

        return box

    def is_document_cut(self, corners, image_shape, threshold=10):
        """문서의 가장자리가 이미지 경계에 너무 가까운지 확인합니다."""
        height, width = image_shape[:2]
        for (x, y) in corners:
            if x <= threshold or x >= width - threshold or y <= threshold or y >= height - threshold:
                return True
        return False

    def warp_perspective(self, image, edges):
        """문서의 가장자리를 기준으로 원근 변형을 적용합니다."""
        pts = edges.reshape(4, 2)
        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect
        maxWidth, maxHeight = image.shape[1], image.shape[0]
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_CUBIC)
        return warped

    def process_images_in_directory(self, directory_path, result_directory, cut_directory):
        """디렉토리 내의 모든 이미지를 처리합니다."""
        if not os.path.exists(result_directory):
            os.makedirs(result_directory)
            print(f"Created directory: {result_directory}")

        if not os.path.exists(cut_directory):
            os.makedirs(cut_directory)
            print(f"Created directory: {cut_directory}")

        cut_documents_count = 0

        for filename in os.listdir(directory_path):
            if filename.lower().endswith(".jpg"):
                image_path = os.path.join(directory_path, filename)
                image = cv2.imread(image_path)
                document_edges = self.find_document_edges(image)
                if document_edges is not None:
                    if not self.is_document_cut(document_edges.reshape(4, 2), image.shape):
                        warped = self.warp_perspective(image, document_edges)
                        base_filename, file_extension = os.path.splitext(filename)
                        save_filename = f"{base_filename}_warped{file_extension}"
                        save_path = os.path.join(result_directory, save_filename)
                        cv2.imwrite(save_path, warped)
                        print(f"Processed and saved: {save_path}")
                    else:
                        cut_documents_count += 1
                        cut_save_path = os.path.join(cut_directory, filename)
                        cv2.imwrite(cut_save_path, image)
                        print(f"Document edges are too close to the image boundary, saved to {cut_save_path}")
                else:
                    print(f"No document edges found for: {filename}")

        print(f"Total cut documents excluded: {cut_documents_count}")


# 메인 실행 부분
if __name__ == "__main__":
    # 디렉토리 경로 설정
    directory_path = 'crops/invoice'
    result_directory = 'crops/warped_invoice_5'
    cut_directory = 'crops/cut_invoice'

    # DocumentProcessor 객체 생성
    processor = DocumentProcessor()

    # 수행 시간 측정 시작
    start_time = time.time()

    # 이미지 처리 및 필터링 실행
    processor.process_images_in_directory(directory_path, result_directory, cut_directory)

    # 수행 시간 측정 종료
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total elapsed time for warping: {elapsed_time:.2f} seconds")