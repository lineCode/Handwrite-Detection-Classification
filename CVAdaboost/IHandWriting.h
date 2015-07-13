#ifndef __HandWritingStudying
#define __HandWritingStudying
#define TrainNum 60000
#include <iostream>
#include <fstream>
#include <queue>
#include <vector>
#include <opencv2\opencv.hpp>

using namespace cv;
using namespace std;

class NumTrainData;

vector<NumTrainData> buffer;
int featureLen = 64;

class NumTrainData {
public:
  NumTrainData() {
    memset(data, 0, sizeof(data));
    result = -1;
  }
public:
  float data[64];
  int result;
};

//函数作用： 获取Region of interest
//参数列表： 
//     src   源图像
//     dst   输出图像
//返 回 值： N/A
void GetROI(Mat& src, Mat& dst) {
  int left = src.cols;
  int right = 0;
  int top = src.rows;
  int bottom = 0;
  // 遍历找到ROI的范围
  for (int i = 0; i < src.rows; i++) {
    for (int j = 0; j < src.cols; j++) {
      if (src.at<uchar>(i, j) > 0) {
        if (j < left) { left = j; }
        if (j > right) { right = j; }
        if (i < top) { top = i; }
        if (i > bottom) { bottom = i; }
      }
    }
  }
  // 计算ROI区域的长宽
  int width = right - left;
  int height = bottom - top;
  int len = (width < height) ? height : width;
  // 建立目标图片并拷贝ROI
  dst = Mat::zeros(len, len, CV_8UC1);
  Rect dstRect((len - width) / 2, (len - height) / 2, width, height);
  Rect srcRect(left, top, width, height);
  Mat dstROI = dst(dstRect);
  Mat srcROI = src(srcRect);
  srcROI.copyTo(dstROI);
}

//函数作用： 翻转buffer的内容
//参数列表： 
//     buf   带交换的buffer指针
//返 回 值： N/A
void swapBuffer(char* buf) {
  char temp;
  temp = *(buf);
  *buf = *(buf + 3);
  *(buf + 3) = temp;
  temp = *(buf + 1);
  *(buf + 1) = *(buf + 2);
  *(buf + 2) = temp;
}

//函数作用： 读入训练数据
//参数列表： 
//     noc   训练样本个数
//返 回 值： N/A
void ReadTrainData(int noc) {
  // 载入MNIST
  const char fileName[] = "res\\train-images.idx3-ubyte";
  const char labelFileName[] = "res\\train-labels.idx1-ubyte";
  ifstream lab_ifs(labelFileName, ios_base::binary);
  ifstream ifs(fileName, ios_base::binary);
  if (ifs.fail() == true) { return; }
  if (lab_ifs.fail() == true) { return; }
  // 获取头部的信息
  char magicNum[4], ccount[4], crows[4], ccols[4];
  ifs.read(magicNum, sizeof(magicNum));
  ifs.read(ccount, sizeof(ccount));
  ifs.read(crows, sizeof(crows));
  ifs.read(ccols, sizeof(ccols));
  int count, rows, cols;
  swapBuffer(ccount);
  swapBuffer(crows);
  swapBuffer(ccols);
  memcpy(&count, ccount, sizeof(count));
  memcpy(&rows, crows, sizeof(rows));
  memcpy(&cols, ccols, sizeof(cols));
  lab_ifs.read(magicNum, sizeof(magicNum));
  lab_ifs.read(ccount, sizeof(ccount));
  // 建立临时图片
  Mat src = Mat::zeros(rows, cols, CV_8UC1);
  Mat temp = Mat::zeros(8, 8, CV_8UC1);
  Mat img, dst;
  char label = 0;
  Scalar templateColor(255, 0, 255);
  NumTrainData rtd;
  int total = 0;
  // 读取训练数据
  while (!ifs.eof()) {
    if (total >= count) { break; }
    total++;
    cout << total << endl;
    // 真实值
    lab_ifs.read(&label, 1);
    label = label + '0';
    // 获得图片的ROI
    ifs.read((char*)src.data, rows * cols);
    GetROI(src, dst);
    rtd.result = label;
    resize(dst, temp, temp.size());
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        rtd.data[i * 8 + j] = temp.at<uchar>(i, j);
      }
    }
    // 放置到buffer里等待训练
    buffer.push_back(rtd);
    noc--;
    if (noc == 0)
      break;
  }
  ifs.close();
  lab_ifs.close();
}

//函数作用： 训练SVM分类器
//参数列表： N/A
//返 回 值： N/A
void TrainSVM() {
  // 载入buffer数据
  cout << "training..." << endl;
  int testCount = buffer.size();
  Mat m = Mat::zeros(1, featureLen, CV_32FC1);
  Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);
  Mat res = Mat::zeros(testCount, 1, CV_32SC1);
  for (int i = 0; i < testCount; i++) {
    NumTrainData td = buffer.at(i);
    memcpy(m.data, td.data, featureLen * sizeof(float));
    normalize(m, m);
    memcpy(data.data + i * featureLen * sizeof(float), m.data, featureLen * sizeof(float));
    res.at<unsigned int>(i, 0) = td.result;
  }
  // 训练SVM
  CvSVM svm;
  CvSVMParams param;
  CvTermCriteria criteria;
  criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
  param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);
  svm.train(data, res, Mat(), Mat(), param);
  // 保存模型
  svm.save("svm_model.xml");
  cout << "model trained and saved" << endl;
}

//函数作用： 用SVM识别手写数字
//参数列表： 
//     qcc   划分好的图像队列
//返 回 值： N/A
void DashSVM(queue<Mat>& qcc) {
  cout << "Read model" << endl;
  CvSVM svm;
  svm.load("svm_model.xml");
  cout << "Read model OK" << endl;
  while (!qcc.empty()) {
    // 获取队列中的待识别图片
    Mat src = qcc.front();
    qcc.pop();
    Mat dst;
    // 取得它的有价值区域
    GetROI(src, dst);
    // 放大以便后面观察
    Mat img = Mat::zeros(dst.rows * 20, dst.cols * 20, CV_8UC3);
    resize(dst, img, img.size());
    Mat temp = Mat::zeros(8, 8, CV_8UC1);
    Mat m = Mat::zeros(1, featureLen, CV_32FC1);
    resize(dst, temp, temp.size());
    for (int i = 0; i < 8; i++) {
      for (int j = 0; j < 8; j++) {
        m.at<float>(0, j + i * 8) = temp.at<uchar>(i, j);
      }
    }
    // SVM识别
    normalize(m, m);
    char ret = (char)svm.predict(m);
    // 输出
    stringstream ss;
    ss << "predict " << ret;
    string text = ss.str();
    cout << "> predict: " << ret << endl;
    putText(img, text, Point(2, 20), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(255, 255, 255), 1);
    putText(img, text, Point(2, 40), FONT_HERSHEY_SIMPLEX, 0.4, cvScalar(0, 0, 0), 1);
    imshow("img", img);
    waitKey();
  }
}

//函数作用： 用SVM识别手写数字
//参数列表： 
//     src   源图像
//       x   开始裁剪的x坐标
//       y   开始裁剪的y坐标
//       w   裁剪区域的宽
//       h   裁剪区域的高
//返 回 值： Mat - 裁剪完毕的子图像
Mat Cutter(Mat& src, int x, int y, int w, int h) {
  Mat dst(Size(w, h), src.type());
  for (int i = 0; i < (int)dst.rows; i++) {
    for (int j = 0; j < (int)dst.step; j++) {
      dst.data[i * dst.step + j] = src.data[(y + i) * src.step + x + j];
    }
  }
  return dst;
}

//函数作用： Adaboost检测手写数字
//参数列表：
//   fname   处理图片的路径
//返 回 值： queue<Mat> - 划分好的图片队列
queue<Mat> DashAdaboost(string fname) {
  queue<Mat> predictQueue;
  cout << "Now begin test process" << endl;
  // 加载级联分类器
  CascadeClassifier cascade;
  String cascadeName = ".\\cascade\\cascade.xml";
  if (!cascade.load(cascadeName)) {
    cerr << "# Exception: load classifier cascade failed!" << endl;
    return predictQueue;
  }
  // 读入源图并二值化
  Mat preim = imread(fname), pgg;
  cvtColor(preim, pgg, CV_BGR2GRAY);
  threshold(pgg, pgg, 127, 255, CV_THRESH_BINARY);
  bitwise_not(pgg, pgg);
  // 保存临时图片
  imwrite("pgg.jpg", pgg);
  Mat image;
  image = imread("pgg.jpg");
  if (image.empty()) {
    cerr << "> Exception: load temp file failed!" << endl;
    return predictQueue;
  }
  Mat gray;
  // 灰度化和直方图均衡化
  cvtColor(image, gray, CV_BGR2GRAY);
  equalizeHist(gray, gray);
  equalizeHist(pgg, pgg);
  // Adaboost检测手写数字
  vector<Rect> rects;
  cascade.detectMultiScale(gray, rects, 1.035, 3, 1, Size(5, 5), Size(38, 38));
  // 圈定ROI，压入队列
  for (vector<Rect>::const_iterator pRect = rects.begin(); pRect != rects.end(); pRect++) {
    rectangle(preim, cvPoint(pRect->x, pRect->y), cvPoint(pRect->x + pRect->width, pRect->y + pRect->height), cvScalar(225, 0, 255));
    predictQueue.push(Cutter(pgg, pRect->x, pRect->y, pRect->width, pRect->height));
  }
  // 写到临时文件
  imshow("Dectection Preview", preim);
  waitKey();
  imwrite("Detected.jpg", preim);
  return predictQueue;
}

//函数作用： 交互器
//参数列表： N/A
//返 回 值： int - 用户选择
int Controller() {
  cout << "Choose the mode: (input the num)" << endl
    << "  Note that the model has already trained." << endl
    << "  Retrain will take at least 30 min." << endl
    << "=====================" << endl
    << "  1. Train" << endl
    << "  2. Test" << endl
    << ">";
  int tp;
  cin >> tp;
  return (tp != 1 && tp != 2) ? 2 : tp;
}

#endif