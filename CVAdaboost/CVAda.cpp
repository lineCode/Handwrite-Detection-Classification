#define SourceImage "src.jpg"
#include "IHandWriting.h"

int main() {
  // �ȴ��û�ѡ��
  if (Controller() == 1) {
    // ��ȡѵ������
    ReadTrainData(TrainNum);
    // ѵ��SVM
    TrainSVM();
  } 
  // Adaboost��⡢SVMʶ��
  DashSVM(DashAdaboost(SourceImage));
  return 0;
}