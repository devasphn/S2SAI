import os

# Disable TensorFlow in Transformers and suppress TF logs
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    # Optionally mute any remaining TensorFlow messages
    from silence_tensorflow import silence_tensorflow
    silence_tensorflow("ERROR")
except ImportError:
    pass

from transformers import pipeline
import numpy as np
import torch

class EmotionClassifier:
    """
    EmotionClassifier wraps a Hugging Face audio-classification pipeline
    for real-time emotion detection on audio arrays.
    """
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the audio emotion classification pipeline.

        Args:
            model_name (str): Hugging Face model identifier for audio classification.
            device (str): Device to run on ('cpu' or 'cuda').
        """
        # Determine device index for pipeline: 0 for CUDA, -1 for CPU
        self.device = 0 if device.startswith('cuda') and torch.cuda.is_available() else -1
        self.classifier = pipeline(
            task='audio-classification',
            model=model_name,
            device=self.device
        )

    def predict(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Predict the predominant emotion label for the given audio segment.

        Args:
            audio (np.ndarray): Audio waveform array normalized to [-1, 1].
            sr (int): Sampling rate of the audio.

        Returns:
            str: Predicted emotion label (e.g., 'happiness', 'sadness').
        """
        inputs = {'array': audio, 'sampling_rate': sr}
        results = self.classifier(inputs)
        # Return the top label
        return results[0]['label']
