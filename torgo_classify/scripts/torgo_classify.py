import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from audio_processing import resample_audio, extract_mfcc, extract_2d_mfcc
from label_utils import get_severity_label
from experiment_config import DATASET_ROOT, TARGET_SR, FEATURE_CACHE_DIR
from log_utils import setup_logger
from models import DNN, CNN, LSTM

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
    """训练和评估 PyTorch 模型"""
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
    
    # 训练和评估模型
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    for epoch in range(600):  # 训练 10 轮
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
    
    # 评估模型
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1)
        accuracy = (y_pred == y_test).float().mean().item()
        report = classification_report(y_test.cpu(), y_pred.cpu())
    
    return accuracy, report


def main():
    # 加载和处理数据
    logger.info("Loading and processing data...")
    
    # 加载 1D MFCC 特征（用于 DNN 和 SVM）
    features_1d, labels = load_and_process_data(use_2d_mfcc=False)
    X_train_1d, X_test_1d, y_train, y_test = train_test_split(features_1d, labels, test_size=0.2, random_state=42)
    
    # 加载 2D MFCC 特征（用于 CNN 和 LSTM）
    features_2d, _ = load_and_process_data(use_2d_mfcc=True)
    if len(features_2d) == 0:
        logger.error("No valid 2D MFCC features found!")
        return
    X_train_2d, X_test_2d, _, _ = train_test_split(features_2d, labels, test_size=0.2, random_state=42)
    
    # 训练和评估 SVM
    logger.info("Training SVM classifier...")
    svm = SVC(kernel='linear')
    svm.fit(X_train_1d, y_train)
    y_pred = svm.predict(X_test_1d)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    logger.info(f"SVM Accuracy: {accuracy}")
    logger.info("SVM Classification Report:\n" + report)
    
    # 保存 SVM 结果
    os.makedirs('../results', exist_ok=True)
    with open('../results/classification_results.csv', 'w') as f:
        f.write(f"Model,Accuracy\n")
        f.write(f"SVM,{accuracy}\n")
    
    # 使用 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 训练和评估 DNN
    try:
        dnn = DNN(input_dim=X_train_1d.shape[1])
        dnn_accuracy, dnn_report = train_and_evaluate_pytorch(dnn, X_train_1d, X_test_1d, y_train, y_test, 'DNN', device)
        with open('../results/classification_results.csv', 'a') as f:
            f.write(f"DNN,{dnn_accuracy}\n")
    except Exception as e:
        logger.error(f"Error in DNN training: {e}")
    
    # 训练和评估 CNN
    try:
        cnn = CNN()
        X_train_cnn = np.expand_dims(X_train_2d, axis=1)  # 添加通道维度
        X_test_cnn = np.expand_dims(X_test_2d, axis=1)
        cnn_accuracy, cnn_report = train_and_evaluate_pytorch(cnn, X_train_cnn, X_test_cnn, y_train, y_test, 'CNN', device)
        with open('../results/classification_results.csv', 'a') as f:
            f.write(f"CNN,{cnn_accuracy}\n")
    except Exception as e:
        logger.error(f"Error in CNN training: {e}")
    
 # 训练和评估 LSTM
    try:
        lstm = LSTM()
        # 转置特征形状为 (180, 13)
        X_train_lstm = np.array([mfcc.T for mfcc in X_train_2d])  # 形状 (样本数, 180, 13)
        X_test_lstm = np.array([mfcc.T for mfcc in X_test_2d])    # 形状 (样本数, 180, 13)
        lstm_accuracy, lstm_report = train_and_evaluate_pytorch(lstm, X_train_lstm, X_test_lstm, y_train, y_test, 'LSTM', device)
        with open('../results/classification_results.csv', 'a') as f:
            f.write(f"LSTM,{lstm_accuracy}\n")
    except Exception as e:
        logger.error(f"Error in LSTM training: {e}")

if __name__ == "__main__":
    main()
