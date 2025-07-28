"""
This module provides functions for analyzing audio files.
"""
import librosa

def load_and_extract_pitch(audio_path: str):
    """
    Loads an audio file and extracts pitch information using librosa.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A tuple containing three numpy arrays:
            - pitches (np.ndarray): An array of fundamental frequencies (F0) in Hertz (Hz)
              for each frame. Values can be `np.nan` if the frame is unvoiced or if
              pitch detection is uncertain.
            - voiced_flags (np.ndarray): A boolean array. `True` for frames where pitched
              (voiced) sound is detected, `False` for frames considered unvoiced.
            - voiced_probs (np.ndarray): An array of probabilities (between 0.0 and 1.0)
              representing the likelihood that each frame is voiced. Higher values
              indicate greater confidence in the presence of a voiced signal.
    
    Raises:
        FileNotFoundError: If the audio file is not found.
        Exception: For other errors during audio loading or processing.
    """
    try:
        y, sr = librosa.load(audio_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    except Exception as e:
        raise Exception(f"Error loading audio file {audio_path}: {e}")

    try:
        pitches, voiced_flags, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    except Exception as e:
        raise Exception(f"Error extracting pitch from {audio_path}: {e}")
        
    return pitches, voiced_flags, voiced_probs
