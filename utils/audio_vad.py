import numpy as np
import webrtcvad

def split_audio(audio: np.ndarray, sr: int, aggressiveness: int = 3) -> list[np.ndarray]:
    """
    Split audio into voiced segments using WebRTC VAD.
    """
    vad = webrtcvad.Vad(aggressiveness)
    frame_ms = 30
    frame_len = int(sr * frame_ms / 1000)
    segments = []
    buffer = []

    for i in range(0, len(audio), frame_len):
        frame = audio[i:i + frame_len]
        if len(frame) < frame_len:
            break
        is_speech = vad.is_speech((frame * 32767).astype('int16').tobytes(), sr)
        if is_speech:
            buffer.extend(frame.tolist())
        else:
            if buffer:
                segments.append(np.array(buffer))
                buffer = []
    if buffer:
        segments.append(np.array(buffer))
    return segments
