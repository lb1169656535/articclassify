# articclassify
构音障碍严重程度的分类实验

相关依赖
pip install -r requirements.txt
## 当前实验结果
![alt text](torgo_classify/results/model_comparison.png)
![alt text](torgo_classify/results/metrics/f1_comparison.png)
![alt text](torgo_classify/results/metrics/precision_comparison.png)
![alt text](torgo_classify/results/metrics/recall_comparison.png)

### 混淆矩阵
![alt text](torgo_classify/results/confusion/confusion_CNN.png)
![alt text](torgo_classify/results/confusion/confusion_DNN.png)
![alt text](torgo_classify/results/confusion/confusion_LSTM.png)
![alt text](torgo_classify/results/confusion/confusion_GRU.png)
![alt text](torgo_classify/results/confusion/confusion_SVM.png)
![alt text](torgo_classify/results/confusion/confusion_XGBoost.png)
![alt text](torgo_classify/results/confusion/confusion_ReNet.png)
![alt text](torgo_classify/results/confusion/confusion_RandomForest.png)