#include <Python.h>
#include <iostream>

int main() {
    // Python 인터프리터 초기화
    Py_Initialize();

    // Python 스크립트 파일 경로 설정
    const char* scriptPath = "C:/Users/user/Desktop/ksn/practice/Release/run_files.py";

    // Python 스크립트 파일 열기
    FILE* file = fopen(scriptPath, "r");
    if (file == nullptr) {
        std::cerr << "파일을 열 수 없습니다: " << scriptPath << std::endl;
        Py_Finalize();  // Python 인터프리터 종료
        return 1;
    }

    // Python 스크립트 실행
    int result = PyRun_SimpleFile(file, scriptPath);
    if (result != 0) {
        std::cerr << "Python 스크립트 실행 중 오류가 발생했습니다." << std::endl;
        PyErr_Print();  // Python 에러 메시지 출력
    }

    // 파일 닫기
    fclose(file);

    // Python 인터프리터 종료
    Py_Finalize();

    return 0;
}

