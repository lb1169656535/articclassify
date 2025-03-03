import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from audio_processing import resample_audio, extract_mfcc
from label_utils import get_severity_label
from experiment_config import DATASET_ROOT, TARGET_SR, FEATURE_CACHE_DIR
from log_utils import setup_logger
import joblib
# 初始化日志记录器
logger = setup_logger()
def load_and_process_data():
    """加载数据并处理音频文件，返回特征和标签"""
    features, labels = [], []
    speakers = ['F01', 'F03', 'F04', 'M01', 'M02', 'M03', 'M04', 'M05']
    
    # 如果特征缓存目录不存在，则创建
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    
    for speaker in speakers:
        speaker_dir = os.path.join(DATASET_ROOT, speaker)
        for session in os.listdir(speaker_dir):
            if session.startswith('Session'):
                wav_dir = os.path.join(speaker_dir, session, 'wav_arrayMic')
                if os.path.exists(wav_dir):
                    for audio_file in os.listdir(wav_dir):
                        if audio_file.endswith('.wav'):
                            audio_path = os.path.join(wav_dir, audio_file)
                            # 检查是否有缓存的 MFCC 特征
                            feature_cache_path = os.path.join(FEATURE_CACHE_DIR, f"{speaker}_{session}_{audio_file}.npy")
                            if os.path.exists(feature_cache_path):
                                mfcc = np.load(feature_cache_path)
                            else:
                                try:
                                    # 采样率统一
                                    resample_audio(audio_path, audio_path, TARGET_SR)
                                    # 特征提取
                                    mfcc = extract_mfcc(audio_path)
                                    # 缓存特征
                                    np.save(feature_cache_path, mfcc)
                                except Exception as e:
                                    logger.error(f"Error processing {audio_path}: {e}")
                                    continue
                            features.append(np.mean(mfcc, axis=0))  # 取每帧特征的均值
                            labels.append(get_severity_label(speaker))
    return np.array(features), np.array(labels)
def main():
    # 加载和处理数据
    logger.info("Loading and processing data...")
    features, labels = load_and_process_data()
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # 训练 SVM 分类器
    logger.info("Training SVM classifier...")
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    
    # 评估性能
    logger.info("Evaluating model...")
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    # 保存结果
    with open('../results/classification_report.txt', 'w') as f:
        f.write(report)
    with open('../results/classification_results.csv', 'w') as f:
        f.write(f"Accuracy,{accuracy}\n")
    
    logger.info(f"Accuracy: {accuracy}")
    logger.info("Classification Report:\n" + report)
if __name__ == "__main__":
    main()
