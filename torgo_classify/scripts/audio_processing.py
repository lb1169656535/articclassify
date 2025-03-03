import numpy as np
import librosa
import soundfile as sf
def resample_audio(input_path, output_path, target_sr=16000):
    """统一采样率"""
    y, sr = librosa.load(input_path, sr=None)
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    sf.write(output_path, y_resampled, target_sr)
def extract_mfcc(audio_path, n_mfcc=13):
    """提取 MFCC 特征"""
    y, sr = librosa.load(audio_path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=int(0.005 * sr), n_fft=int(0.025 * sr))
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    features = np.vstack([mfcc, delta_mfcc, delta2_mfcc])
    return features.T  # 转置为 (帧数, 39)
