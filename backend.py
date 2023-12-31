import torch
import librosa
import numpy as np
import pyaudio
import wave
import os
import threading
from librosa.sequence import dtw
from PyQt5.QtCore import QObject, pyqtSignal


class RecordingThread(QObject):
    finished = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.audio_path = "output.wav"  # Define the path to save the recorded audio
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        duration = 5
        frames = []
        if self.stream is None:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )

            for _ in range(0, int(RATE / CHUNK * duration)):
                data = self.stream.read(CHUNK)
                frames.append(data)

            # Stop and close the audio stream
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            wf = wave.open(self.audio_path, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(frames))
            wf.close()

            self.finished.emit()
        else:
            print("Stream is already active")

    def stop_recording(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()

            wf = wave.open(self.audio_path, "wb")
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(44100)
            wf.writeframes(b"".join(self.frames))
            wf.close()

            self.frames = []  # Clear frames after saving
            self.stream = None
            self.finished.emit()
        else:
            print("No active stream to stop")

    def cleanup(self):
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()


class Processing:
    def __init__(self):
        self.audio_path = ""

    def extract_combined_features(self):
        x, sample_rate = librosa.load(self.audio_path)
        normalized_x = librosa.util.normalize(x)
        # MFCC extraction
        mfcc = np.mean(
            librosa.feature.mfcc(y=normalized_x, sr=sample_rate, n_mfcc=50), axis=1
        )

        # Chroma Features extraction
        chroma = np.mean(
            librosa.feature.chroma_stft(y=normalized_x, sr=sample_rate), axis=1
        )

        # Zero Crossing Rate (ZCR) calculation
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=normalized_x), axis=1)

        # Spectral Centroid
        spectrogram = np.mean((np.abs(librosa.stft(normalized_x))), axis=1)

        # Spectral Contrast
        contrast = np.mean(
            librosa.feature.spectral_contrast(y=normalized_x, sr=sample_rate), axis=1
        )

        # Combine features
        combined_features = np.concatenate([mfcc, zcr, contrast, spectrogram, chroma])

        return combined_features


class Audio:
    def __init__(self):
        pass


class recognition_app:
    def __init__(self, ui):
        self.ui = ui
        self.recording_thread = RecordingThread()
        self.recording_thread.finished.connect(self.on_finished)

        self.ui.rec_btn.clicked.connect(self.start_recording)
        self.ui.stop_btn.clicked.connect(self.stop_recording)

    def start_recording(self):
        threading.Thread(target=self.recording_thread.start_recording).start()

    def stop_recording(self):
        self.recording_thread.stop_recording()

    def on_finished(self):
        print("Recording complete.")
        pass
