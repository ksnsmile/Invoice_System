#include <Python.h>
#include <iostream>
#include <string>

int main() {
    // Conda 환경의 Python 경로 설정
    std::wstring pythonHome = L"C:\\Users\\user\\Anaconda3\\envs\\nami";

    // Py_SetPythonHome deprecated 경고를 피하기 위해 환경 변수로 설정
    _putenv_s("PYTHONHOME", "C:\\Users\\user\\Anaconda3\\envs\\nami");

    // Python 인터프리터 초기화
    Py_Initialize();

    // Python 인터프리터가 초기화되었는지 확인
    if (!Py_IsInitialized()) {
        std::cerr << "Python initialization failed!" << std::endl;
        return 1;
    }

    // Python 코드 실행
    int ret = PyRun_SimpleString("print('Hello from Python in Conda environment!')");

    // 오류 확인
    if (ret != 0) {
        std::cerr << "Error running Python code!" << std::endl;
        return 1;
    }

    // Python 인터프리터 종료
    Py_Finalize();
    return 0;
}


