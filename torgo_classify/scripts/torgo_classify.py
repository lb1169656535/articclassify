import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
from audio_processing import resample_audio, extract_mfcc, extract_2d_mfcc
from label_utils import get_severity_label
from experiment_config import DATASET_ROOT, TARGET_SR, FEATURE_CACHE_DIR
from log_utils import setup_logger
from models import DNN, CNN, LSTM

from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from models import ResNet, GRU 

import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # 添加当前目录到PATH
from utils.visualization import plot_model_comparison,plot_confusion_matrices,plot_metrics


# 初始化日志记录器
logger = setup_logger()


def load_and_process_data(use_2d_mfcc=False):
    """加载数据并处理音频文件，返回特征和标签"""
    features, labels = [], []
    speakers = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']
    
    FEATURE_CACHE_DIR_1D = os.path.join(FEATURE_CACHE_DIR, '1d_mfcc')
    FEATURE_CACHE_DIR_2D = os.path.join(FEATURE_CACHE_DIR, '2d_mfcc')
    # 创建缓存文件夹
    os.makedirs(FEATURE_CACHE_DIR_1D, exist_ok=True)
    os.makedirs(FEATURE_CACHE_DIR_2D, exist_ok=True)

    

    for speaker in speakers:
        speaker_dir = os.path.join(DATASET_ROOT, speaker)
        for session in os.listdir(speaker_dir):
            if session.startswith('Session'):
                wav_dir = os.path.join(speaker_dir, session, 'wav_arrayMic')
                if os.path.exists(wav_dir):
                    for audio_file in os.listdir(wav_dir):
                        if audio_file.endswith('.wav'):
                            audio_path = os.path.join(wav_dir, audio_file)
                            
                            # 根据特征类型选择缓存路径
                            if use_2d_mfcc:
                                feature_cache_path = os.path.join(FEATURE_CACHE_DIR_2D, f"{speaker}_{session}_{audio_file}.npy")
                            else:
                                feature_cache_path = os.path.join(FEATURE_CACHE_DIR_1D, f"{speaker}_{session}_{audio_file}.npy")
                            
                            # 检查是否有缓存的 MFCC 特征
                            if os.path.exists(feature_cache_path):
                                mfcc = np.load(feature_cache_path)
                            else:
                                try:
                                    # 采样率统一
                                    resample_audio(audio_path, audio_path, TARGET_SR)
                                    # 特征提取
                                    if use_2d_mfcc:
                                        mfcc = extract_2d_mfcc(audio_path)  # 提取二维 MFCC
                                    else:
                                        mfcc = extract_mfcc(audio_path)  # 提取 1D MFCC
                                    # 缓存特征
                                    np.save(feature_cache_path, mfcc)
                                except Exception as e:
                                    logger.error(f"Error processing {audio_path}: {e}")
                                    continue
                            
                            # 确保特征形状正确
                            if use_2d_mfcc:
                                if mfcc.shape != (13, 180):
                                    logger.error(f"Invalid shape for 2D MFCC: {mfcc.shape}")
                                    continue
                                features.append(mfcc)  # 直接存储二维 MFCC
                            else:
                                features.append(np.mean(mfcc, axis=0))  # 取每帧特征的均值
                            labels.append(get_severity_label(speaker))
    return np.array(features), np.array(labels)

def train_and_evaluate_pytorch(model, X_train, X_test, y_train, y_test, model_name, device):
    """训练和评估 PyTorch 模型（返回准确率、分类报告、混淆矩阵）"""
    # 将模型移动到设备
    model = model.to(device)
    
    # 确保输入数据是数值类型
    X_train = np.array(X_train, dtype=np.float32)
    X_test = np.array(X_test, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    y_test = np.array(y_test, dtype=np.int32)
    
    # 转换为 PyTorch 张量并移动到设备
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test = torch.tensor(y_test, dtype=torch.long).to(device)
    
    # 训练配置
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    model.train()
    for epoch in range(600):
        if (epoch+1) % 100 == 0:  # 每100轮输出进度
            print(f"Training {model_name} | Epoch {epoch+1}/600")
            
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # 评估模式
    model.eval()
    with torch.no_grad():
        # 获取预测结果
        test_outputs = model(X_test)
        y_pred = test_outputs.argmax(dim=1)
        
        # 计算准确率
        accuracy = (y_pred == y_test).float().mean().item()
        
        # 转换为numpy数组
        y_test_np = y_test.cpu().numpy()
        y_pred_np = y_pred.cpu().numpy()
        
        # 生成分类报告和混淆矩阵
        report_dict = classification_report(
            y_test_np, y_pred_np,
            target_names=['Very Low', 'Low', 'Medium'],
            output_dict=True
        )
        confusion = confusion_matrix(y_test_np, y_pred_np)
    
    return accuracy, report_dict, confusion

# 在torgo_classify.py中添加以下辅助函数
def save_detailed_metrics(report_dict, model_name, output_path):
    """保存详细指标到CSV"""
    classes = ['Very Low', 'Low', 'Medium']
    rows = []
    for class_name in classes:
        metrics = report_dict[class_name]
        rows.append({
            'Model': model_name,
            'Class': class_name,
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1': metrics['f1-score'],
            'Support': metrics['support']
        })
    df = pd.DataFrame(rows)
    header = not os.path.exists(output_path)
    df.to_csv(output_path, mode='a', header=header, index=False)

def save_confusion_matrix(confusion, model_name, output_dir):
    """保存混淆矩阵到CSV"""
    classes = ['Very Low', 'Low', 'Medium']
    df = pd.DataFrame(confusion, 
                     index=pd.Index(classes, name='True'),
                     columns=pd.Index(classes, name='Predicted'))
    df.to_csv(os.path.join(output_dir, f'confusion_matrix_{model_name}.csv'))

def main():
    # 初始化结果文件
    results_dir = '../results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 清空或创建结果文件
    with open(os.path.join(results_dir, 'classification_results.csv'), 'w') as f:
        f.write("Model,Accuracy\n")
    with open(os.path.join(results_dir, 'detailed_metrics.csv'), 'w') as f:
        f.write("Model,Class,Precision,Recall,F1,Support\n")
    
    # 加载和处理数据
    logger.info("Loading and processing data...")
    
    # 加载 1D MFCC 特征
    features_1d, labels = load_and_process_data(use_2d_mfcc=False)
    X_train_1d, X_test_1d, y_train, y_test = train_test_split(
        features_1d, labels, test_size=0.2, random_state=42
    )
    
    # 加载 2D MFCC 特征
    features_2d, _ = load_and_process_data(use_2d_mfcc=True)
    if len(features_2d) == 0:
        logger.error("No valid 2D MFCC features found!")
        return
    X_train_2d, X_test_2d, _, _ = train_test_split(
        features_2d, labels, test_size=0.2, random_state=42
    )
    
    # 定义设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ==================== 传统机器学习模型 ====================
    def evaluate_sklearn_model(model, model_name, X_train, X_test):
        """通用sklearn模型评估函数"""
        try:
            logger.info(f"Training {model_name} classifier...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # 计算指标
            accuracy = accuracy_score(y_test, y_pred)
            report_dict = classification_report(
                y_test, y_pred,
                target_names=['Very Low', 'Low', 'Medium'],
                output_dict=True
            )
            confusion = confusion_matrix(y_test, y_pred)
            
            # 保存结果
            with open('../results/classification_results.csv', 'a') as f:
                f.write(f"{model_name},{accuracy:.4f}\n")
            save_detailed_metrics(report_dict, model_name, '../results/detailed_metrics.csv')
            save_confusion_matrix(confusion, model_name, '../results')
            
            logger.info(f"{model_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"Error in {model_name} training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # SVM
    evaluate_sklearn_model(SVC(kernel='linear'), "SVM", X_train_1d, X_test_1d)
    
    # RandomForest
    evaluate_sklearn_model(
        RandomForestClassifier(n_estimators=100, random_state=42),
        "RandomForest",
        X_train_1d,
        X_test_1d
    )
    
    # XGBoost
    evaluate_sklearn_model(
        xgb.XGBClassifier(
            objective='multi:softmax',
            num_class=3,  # 修正为3分类
            n_estimators=100,
            learning_rate=0.1
        ),
        "XGBoost",
        X_train_1d,
        X_test_1d
    )

    # ==================== PyTorch 深度学习模型 ====================
    def evaluate_pytorch_model(model_class, model_name, input_type, reshape_fn=None):
        """通用PyTorch模型评估函数"""
        try:
            logger.info(f"Initializing {model_name}...")
            
            # 根据输入类型选择数据
            if input_type == '1d':
                X_train = X_train_1d
                X_test = X_test_1d
            elif input_type == '2d':
                X_train = X_train_2d
                X_test = X_test_2d
            
            # 数据预处理
            if reshape_fn:
                X_train = reshape_fn(X_train)
                X_test = reshape_fn(X_test)
            
            # 初始化模型
            if model_class == DNN:
                model = model_class(input_dim=X_train.shape[1])
            else:
                model = model_class()
            
            # 训练评估
            accuracy, report_dict, confusion = train_and_evaluate_pytorch(
                model, X_train, X_test, y_train, y_test, model_name, device
            )
            
            # 保存结果
            with open('../results/classification_results.csv', 'a') as f:
                f.write(f"{model_name},{accuracy:.4f}\n")
            save_detailed_metrics(report_dict, model_name, '../results/detailed_metrics.csv')
            save_confusion_matrix(confusion, model_name, '../results')
            
            logger.info(f"{model_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"Error in {model_name} training: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # DNN
    evaluate_pytorch_model(DNN, "DNN", '1d')

    # CNN
    evaluate_pytorch_model(
        CNN, "CNN", '2d',
        reshape_fn=lambda x: np.expand_dims(x, axis=1)  # 添加通道维度
    )

    # LSTM
    evaluate_pytorch_model(
        LSTM, "LSTM", '2d',
        reshape_fn=lambda x: np.array([mfcc.T for mfcc in x])  # 转置为 (seq_len, features)
    )

    # ResNet
    evaluate_pytorch_model(
        ResNet, "ResNet", '2d',
        reshape_fn=lambda x: np.expand_dims(x, axis=1)  # 添加通道维度
    )

    # GRU
    evaluate_pytorch_model(
        GRU, "GRU", '2d',
        reshape_fn=lambda x: np.array([mfcc.T for mfcc in x])  # 转置为 (seq_len, features)
    )

    # 生成可视化结果
    logger.info("Generating visualizations...")
    try:
        plot_model_comparison(
            os.path.abspath('../results/classification_results.csv'),
            os.path.abspath('../results/model_comparison.png')
        )
        plot_metrics(os.path.abspath('../results/detailed_metrics.csv'))
        plot_confusion_matrices(os.path.abspath('../results'))
        logger.info("Visualization generation completed")
    except Exception as e:
        logger.error(f"Visualization generation failed: {str(e)}")


if __name__ == "__main__":
    main()
