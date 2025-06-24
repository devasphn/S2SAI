import numpy as np
import librosa

def preprocess_audio(audio: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    Resample and normalize an audio array to the target sample rate.
    """
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    # Normalize amplitude to [-1, 1]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio
