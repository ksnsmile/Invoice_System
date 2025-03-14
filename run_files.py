# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# run.py

from yolov5.detect import Main_YOLO, parse_opt
from threading import Thread

def main():
    


      print("Running YOLO...")
     
      opt = parse_opt()
      opt.weights = r"C:\Users\user\Desktop\ksn\Invoice_System\yolov5\best.pt"
      opt.source = r"C:\Users\user\Desktop\ksn\Invoice_System\yolov5\images"
      opt.imgsz = (1536,1536)
      opt.conf_thres = 0.9
      opt.save_crop = True
      
      
     
      Main_YOLO(opt)
      
      print("완료")      

#%%    
if __name__ == "__main__":
    main()
#%%