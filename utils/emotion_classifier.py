import os
import numpy as np
import tempfile
import soundfile as sf

# Disable TensorFlow in Transformers if installed
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from speechbrain.inference.interfaces import foreign_class

class EmotionClassifier:
    """
    EmotionClassifier using SpeechBrain’s wav2vec2-IEMOCAP model.
    """

    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Args:
          model_path: Path to the folder containing SpeechBrain checkpoints
          device: 'cpu' or 'cuda'
        """
        self.device = device
        self.classifier = foreign_class(
            source=model_path,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device}
        )

    def predict(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Predicts emotion label from a raw audio numpy array.

        Args:
          audio: 1D numpy array of audio samples
          sr: Sampling rate of the audio

        Returns:
          Predicted emotion label string
        """
        # Save audio to temporary WAV file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            sf.write(tmp.name, audio, sr)
            # Perform classification
            out_prob, score, index, text_lab = self.classifier.classify_file(tmp.name)

        # text_lab can be a list or string
        label = text_lab[0] if isinstance(text_lab, list) else text_lab
        return label
