from transformers import pipeline
import torch
import numpy as np

class EmotionClassifier:
    def __init__(self, model_name: str, device: str = 'cpu'):
        """
        Initialize the emotion classification pipeline.
        """
        self.device = 0 if device.startswith('cuda') else -1
        self.classifier = pipeline(
            task='audio-classification',
            model=model_name,
            device=self.device
        )

    def predict(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Predict emotion label for the given audio segment.
        """
        inputs = {'array': audio, 'sampling_rate': sr}
        results = self.classifier(inputs)
        return results[0]['label']
