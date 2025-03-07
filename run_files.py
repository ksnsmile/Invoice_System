# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# run.py
from Check import Main_check
from Preprocessing import Main_preprocessing
from yolov5.detect import Main_YOLO, parse_opt

def main():
    


      print("Running YOLO...")
     
      opt = parse_opt()
      opt.weights = r"C:\Users\user\Desktop\ksn\invocie_system_2\yolov5\best.pt"
      opt.source = r"C:\Users\user\Desktop\ksn\invocie_system_2\yolov5\images"
      opt.imgsz = (1536,1536)
      opt.conf_thres = 0.9
      opt.save_crop = True
      
     
      Main_YOLO(opt)
      print("YOLO completed")
 
     
      print("Running preprocessing...")
      Main_preprocessing()  # preprocessing.py의 main 함수 실행
      print("preporcessing completed")

     

      print("Running check...")
      Main_check()  # check.py의 main 함수 실행
      print("check completed")

    
if __name__ == "__main__":
    main()