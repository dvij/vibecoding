"""
This module provides functions for analyzing audio files.
"""
import librosa


def freq_to_cents(freq, reference_freq):
    """Convert frequency ratio to cents (1200 cents = 1 octave)"""
    return 1200 * np.log2(freq / reference_freq)

def cents_to_note_name(cents):
    """Convert cents from reference to note name"""
    # Chromatic scale starting from reference (0 cents)
    note_names = ['Unison', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']
    
    # Convert to semitones (100 cents = 1 semitone)
    semitones = round(cents / 100)
    octaves = semitones // 12
    note_idx = semitones % 12
    
    if octaves == 0:
        return note_names[note_idx]
    else:
        return f"{note_names[note_idx]} (+{octaves} oct)" if octaves > 0 else f"{note_names[note_idx]} ({octaves} oct)"


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
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        drone_section = f0[:int(2 * sr / 512)]  # First 2 seconds
        drone_freq = np.median(drone_section[voiced_flag[:len(drone_section)]])
        
        # Create arrays with same shape as f0
        relative_cents = np.full_like(f0, np.nan)  # Array of cents, NaN for unvoiced
        relative_notes = np.full(f0.shape, None, dtype=object)  # Array of note names, None for unvoiced
        
        # Calculate relative intervals for all frames
        for i, (freq, is_voiced) in enumerate(zip(f0, voiced_flag)):
            if is_voiced and not np.isnan(freq):
                cents = freq_to_cents(freq, drone_freq)
                note_name = cents_to_note_name(cents)
                relative_cents[i] = cents
                relative_notes[i] = note_name
                
        return relative_notes, voiced_flags, voiced_probs
