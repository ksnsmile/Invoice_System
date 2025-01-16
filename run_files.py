# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# run.py
from check import Main_check
from preprocessing import Main_preprocessing


def main():
    print("Running preprocessing...")
    Main_preprocessing()  # preprocessing.py의 main 함수 실행
    print("preporcessing completed")
    
    print("Running check...")
    Main_check()  # check.py의 main 함수 실행
    print("check completed")
if __name__ == "__main__":
    main()