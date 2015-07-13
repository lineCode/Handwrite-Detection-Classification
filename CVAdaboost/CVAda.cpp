#define SourceImage "src.jpg"
#include "IHandWriting.h"

int main() {
  // 等待用户选择
  if (Controller() == 1) {
    // 读取训练数据
    ReadTrainData(TrainNum);
    // 训练SVM
    TrainSVM();
  } 
  // Adaboost检测、SVM识别
  DashSVM(DashAdaboost(SourceImage));
  return 0;
}