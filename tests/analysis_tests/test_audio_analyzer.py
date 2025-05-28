"""
Tests for the audio_analyzer module.
"""
import unittest
import os
import numpy as np
from src.analysis.audio_analyzer import load_and_extract_pitch

class TestAudioAnalyzer(unittest.TestCase):
    """Test suite for audio analysis functions."""

    def setUp(self):
        """Set up for test methods."""
        self.sample_audio_path = "data/sample_audio/sine_440hz.wav"
        # Ensure the sample audio file exists
        if not os.path.exists(self.sample_audio_path):
            # This is a fallback, ideally the file is created by a separate script
            # For now, we'll raise an error if it's missing as the test depends on it.
            raise FileNotFoundError(
                f"Sample audio file not found: {self.sample_audio_path}. "
                "Please generate it before running tests."
            )

    def test_pitch_extraction_on_sine_wave(self):
        """
        Tests the load_and_extract_pitch function with a 440Hz sine wave.
        """
        pitches, voiced_flags, voiced_probs = load_and_extract_pitch(self.sample_audio_path)

        self.assertTrue(len(pitches) > 0, "Pitches array should not be empty.")
        self.assertTrue(len(voiced_flags) > 0, "Voiced flags array should not be empty.")
        self.assertTrue(len(voiced_probs) > 0, "Voiced probabilities array should not be empty.")

        self.assertEqual(len(pitches), len(voiced_flags), "Pitches and voiced_flags arrays should have the same length.")
        self.assertEqual(len(pitches), len(voiced_probs), "Pitches and voiced_probs arrays should have the same length.")

        # Check that there are voiced frames
        self.assertTrue(np.any(voiced_flags), "No voiced frames detected in the sine wave.")

        # Extract pitches where the frame is voiced
        voiced_pitches = pitches[voiced_flags]
        
        self.assertTrue(len(voiced_pitches) > 0, "No voiced pitches found.")

        # Check that a significant portion of voiced pitches are close to 440 Hz
        target_freq = 440.0
        tolerance = 5.0  # Hz
        
        correct_pitch_count = 0
        for pitch in voiced_pitches:
            if abs(pitch - target_freq) < tolerance:
                correct_pitch_count += 1
        
        # We expect at least 80% of the voiced frames to have the correct pitch
        # This threshold might need adjustment based on librosa's pYIN accuracy for clean tones
        proportion_correct = correct_pitch_count / len(voiced_pitches)
        self.assertTrue(proportion_correct >= 0.80, 
                        f"Expected at least 80% of voiced pitches to be around {target_freq}Hz, "
                        f"but got {proportion_correct*100:.2f}%. "
                        f"Detected pitches: {voiced_pitches[:20]}...") # Log some pitches

    def test_load_non_existent_file(self):
        """
        Tests that load_and_extract_pitch raises FileNotFoundError for a non-existent file.
        """
        with self.assertRaises(FileNotFoundError):
            load_and_extract_pitch("non_existent_audio_file.wav")

if __name__ == '__main__':
    unittest.main()
