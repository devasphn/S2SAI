import os
import numpy as np

# Disable TensorFlow in Transformers
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from speechbrain.inference.interfaces import foreign_class

class EmotionClassifier:
    """
    SpeechBrain-based emotion classifier using the wav2vec2-IEMOCAP model.
    """
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the SpeechBrain emotion classification model.
        
        Args:
            model_path (str): Path to the SpeechBrain model directory
            device (str): Device to run on ('cpu' or 'cuda')
        """
        self.device = device
        self.model_path = model_path
        
        # Load using SpeechBrain's foreign_class function
        self.classifier = foreign_class(
            source=model_path,
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": device}
        )
    
    def predict(self, audio: np.ndarray, sr: int = 16000) -> str:
        """
        Predict emotion from audio array.
        
        Args:
            audio (np.ndarray): Audio waveform
            sr (int): Sample rate
            
        Returns:
            str: Predicted emotion label
        """
        # Save audio temporarily for SpeechBrain processing
        import tempfile
        import soundfile as sf
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            sf.write(tmp_file.name, audio, sr)
            
            # Use SpeechBrain's classify_file method
            out_prob, score, index, text_lab = self.classifier.classify_file(tmp_file.name)
            
            # Clean up temporary file
            os.unlink(tmp_file.name)
            
            return text_lab[0] if isinstance(text_lab, list) else text_lab
