import torch
import librosa
import numpy as np
import pyaudio
import wave
import matplotlib.pyplot as plt
import os
from PyQt6.QtWidgets import QLabel
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from sklearn.metrics.pairwise import cosine_similarity
from librosa.sequence import dtw
# Define paths for passcode records and user records
passcode_records_paths = ["Open the door", "Unlock middle gate", "Grant me access", "None"]

users_records_paths = [
   "Abdallah Magdy",
    "Abdelrahman Emad",
    "Mahmoud Mohamed",
    "Mohamed Ibrahim",
    "Muhammed Alaa",
    "Ziad Hossam",
]

# Define the name of the live audio file and check available device (CPU/GPU)
live_audio_file = "live_audio.wav"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Function to record audio for a specified duration and save it to a file
def record_audio(outputLabel: QLabel, duration=2, output_file="recorded_audio.wav"):
    """
    Records audio from the microphone for a specified duration and saves it to a WAV file.

    Parameters:
    - duration (int, optional): Duration of the recording in seconds (default is 2 seconds).
    - output_file (str, optional): Name of the output WAV file to save the recorded audio
                                   (default is "recorded_audio.wav").

    Note:
    - This function uses PyAudio to access the microphone and record audio.
    - The recorded audio will be saved as a WAV file with the specified file name.

    Example usage:
    - To record 5 seconds of audio and save it as "my_audio.wav":
      record_audio(duration=5, output_file="my_audio.wav")
    """
    # Define audio parameters
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100

    # Initialize PyAudio object
    p = pyaudio.PyAudio()

    # Open audio stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    # Inform the user about the recording process
    print(f"Recording audio for {duration} seconds. Speak into the microphone.")
    outputLabel.setText("Recording...")

    frames = []

    # Record audio for the specified duration
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    # Inform the user that the recording is complete
    print("Recording complete.")
    outputLabel.setText("Recording complete, processing your audio...")

    # Stop and close the audio stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(output_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def extract_features(audio_file):
    """
    Extracts audio features from an audio file using librosa.

    Parameters:
    - audio_file (str): Path to the audio file.

    Returns:
    - all_features (numpy.ndarray): Concatenated array of audio features including:
        - Chroma feature (12 different pitch classes)
        - Spectral contrast (difference in amplitude between peaks and valleys)
        - Zero-crossing rate (rate of zero crossings in the signal)
        - Mel-frequency cepstral coefficients (MFCCs)
        - Spectrogram

    Note:
    - This function uses the librosa library to extract audio features from the given audio file.
    - The output 'all_features' is a concatenated array of different audio features.
    """

    # Load the audio file using librosa
    y, sr = librosa.load(audio_file)
    librosa.util.normalize(y)
    # Extract various audio features using librosa functions
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    spectrogram = np.abs(librosa.stft(y))

    # Concatenate the mean values of different features along axis=1
    all_features = np.concatenate([
        np.mean(chroma, axis=1),
        np.mean(spectrogram, axis=1),
        np.mean(spectral_contrast, axis=1),
        np.mean(zero_crossing_rate, axis=1),
        np.mean(mfccs, axis=1)
    ])

    return all_features

#should return the input spectrogram
def plot_spectrogram(audio_file=live_audio_file, title="Spectrogram"):
    """
    Plots a spectrogram for the given audio signal.

    Parameters:
    - y (np.ndarray): Audio time series.
    - sr (number > 0): Sampling rate of the audio.
    - title (str): Title for the spectrogram plot (default: "Spectrogram").

    Returns:
    - None

    Note:
    - This function computes and displays the mel spectrogram of the input audio signal 'y'
      with a sampling rate 'sr'.
    - It uses librosa's melspectrogram function to compute the spectrogram
      and displays it using matplotlib.
    - 'y' should be a numpy array representing the audio signal.
    - 'sr' should be a positive number indicating the sampling rate in Hz.
    - 'title' is an optional parameter to set the title of the plot.

    Example usage:
    - audio_file = "path_to_your_audio_file.wav"
      y, sr = librosa.load(audio_file)
      plot_spectrogram(y, sr, title="Spectrogram of Audio File")
    """
    y, sr = librosa.load(audio_file)
    librosa.util.normalize(y)
    # Compute the spectrogram
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Convert the power spectrogram to decibels
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)

    # Create a new figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the spectrogram
    img = librosa.display.specshow(spectrogram_db, sr=sr, x_axis='time', y_axis='mel', ax=ax)
    plt.colorbar(img, format='%+2.0f dB')  # Specify the mappable object for the colorbar
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency (Mel scale)')

    # Create a FigureCanvas from the figure
    canvas = FigureCanvas(fig)
    
    return canvas


def compare_audio_files(features1, features2):
    """
    Compares the audio features of two audio files using the correlation coefficient.

    Parameters:
    - features1 (numpy.ndarray): Array of audio features for the first audio file.
    - features2 (numpy.ndarray): Array of audio features for the second audio file.

    Returns:
    - correlation_coefficient (float): Correlation coefficient between the audio features.

    Note:
    - This function calculates the correlation coefficient between the provided audio features.
    - It uses numpy's corrcoef function to compute the correlation matrix and extract the coefficient.
    """

    # Compute the correlation matrix of the provided features
    # correlation_matrix = np.corrcoef(features1, features2)
    similarity = cosine_similarity(features1.reshape(1, -1), features2.reshape(1, -1))
  
    distance, path = dtw(features1.T, features2.T, subseq=True)
    # Extract the correlation coefficient from the correlation matrix
    # correlation_coefficient = correlation_matrix[0, 1]
    return np.sum(distance)


def generate_spectrogram(audio_file):
    """
    Generates the spectrogram of an audio file and converts it into a PyTorch tensor.

    Parameters:
    - audio_file (str): Path to the audio file.

    Returns:
    - spectrogram_tensor (torch.Tensor): Spectrogram of the audio file as a PyTorch tensor.

    Note:
    - This function computes the spectrogram of the provided audio file using librosa.
    - The spectrogram is converted into a PyTorch tensor before being returned.
    """

    # Load the audio file using librosa
    y, sr = librosa.load(audio_file)
    librosa.util.normalize(y)
    # Compute the spectrogram of the audio file
    spectrogram = np.abs(librosa.stft(y))

    # Convert the spectrogram (NumPy array) to a PyTorch tensor
    spectrogram_tensor = torch.from_numpy(spectrogram)

    return spectrogram_tensor


def voice_access_control(outputLabel: QLabel):
    """
    Performs voice access control by comparing live audio with stored passcode records.

    Parameters:
    - passcode_records_paths (list): List of paths containing passcode records.

    Returns:
    - sentence_with_max_correlation (str): Sentence associated with the highest similarity score.
    - probabilities (torch.Tensor): Probabilities corresponding to each passcode's similarity score.

    Note:
    - This function records audio from the microphone, generates its spectrogram,
      compares it with stored passcode records' spectrograms, and computes similarity scores.
    - Passcode records paths must be provided as a list to compare with the live audio.

    Example usage:
    - passcode_records = ["path/to/passcode_records_1", "path/to/passcode_records_2"]
      sentence, probs = voice_access_control(passcode_records)
    """
    # Step 1: Record audio from the microphone
    record_audio(outputLabel=outputLabel, output_file=live_audio_file)
    # Generate spectrogram for the recorded live audio
    live_audio_spectrogram = generate_spectrogram(live_audio_file)
    passcodes_similarity_scores = {}

    # Loop through passcode records paths
    for path in passcode_records_paths:
        # List all records in the current passcode path
        records = os.listdir(path)

        # Generate spectrograms for passcode records
        records_spectrogram = [generate_spectrogram(os.path.join(path, record)) for record in records]

        # Convert live audio spectrogram to a PyTorch tensor and move to device
        live_audio_spectrogram = live_audio_spectrogram.to(device=device, dtype=torch.float32)

        # Convert records' spectrograms to PyTorch tensors and move to device
        records_spectrogram = [spec.to(device=device, dtype=torch.float32) for spec in records_spectrogram]

        # Calculate similarity scores using convolutional operations
        similarity_scores = [
            torch.max(torch.nn.functional.conv2d(live_audio_spectrogram.unsqueeze(0),
                                                 spec.unsqueeze(0).unsqueeze(0), padding='same'))
            for spec in records_spectrogram
        ]

        # Calculate average similarity score for each passcode
        passcodes_similarity_scores[os.path.basename(path)] = sum(similarity_scores) / len(records)

        
    # Normalize similarity scores to probabilities using softmax
    values = torch.tensor(list(passcodes_similarity_scores.values()), dtype=torch.float32, device=device)
    probabilities = torch.nn.functional.softmax(values / torch.max(values), dim=0)
    probabilities /= (torch.max(probabilities) + .02)
    # Print probabilities, similarity scores, and the passcode with the highest score
    max_similarity_score = max(passcodes_similarity_scores.values())
    sentence_with_max_correlation = max(passcodes_similarity_scores, key=passcodes_similarity_scores.get)

    return sentence_with_max_correlation, probabilities


def security_voice_fingerprint(granted_users: list, outputLabel: QLabel):
    """
    Implements voice-based security access control using audio features and stored references.

    Parameters:
    - granted_users (list): List of authorized users.

    Returns:
    - granted_access (bool): True if access is granted, False otherwise.
    - probabilities (torch.Tensor): Probabilities corresponding to each user's similarity score.

    Note:
    - This function extracts features from live audio, compares them with stored reference audio features,
      calculates similarity scores, and grants access based on a predefined threshold for authorized users.
    - Ensure that the 'live_audio_file', 'users_records_paths', 'extract_features', 'compare_audio_files',
      and 'voice_access_control' functions are appropriately defined and available in the scope.

    Example usage:
    - authorized_users = ["user1", "user2"]
      access_granted, probs = security_voice_fingerprint(authorized_users)
    """
    sentence, sentences_probabilities = voice_access_control(outputLabel)
    passcode_features = extract_features(live_audio_file)
    passcode_features = torch.from_numpy(passcode_features)
    passcode_features = passcode_features.to(device=device, dtype=torch.float32)
    threshold = .95
    users_similarity_scores = {}

    # Step 3: Compare features with each stored reference
    for user_path in users_records_paths:
        user_records = os.listdir(user_path)
        user_name = os.path.basename(user_path)

        # Extract features for each reference audio
        reference_features = np.array([extract_features(os.path.join(user_path, record)) for record in user_records])

        # Reshape the reference features for computation
        reference_features.reshape(-1, reference_features.shape[0], reference_features[0].shape[0])

        # Compute mean features from reference recordings
        mean_reference_features = np.sum(reference_features.T, axis=1) / len(user_records)
        mean_reference_features = torch.from_numpy(mean_reference_features)
        mean_reference_features = mean_reference_features.to(device=device, dtype=torch.float32)

        # Calculate similarity score between passcode and mean reference features
        similarity_score = torch.max(torch.nn.functional.conv1d(mean_reference_features.unsqueeze(0),
                                                 passcode_features.unsqueeze(0).unsqueeze(0), padding='same'))
        users_similarity_scores[user_name] = similarity_score
        print(f"Similarity score with {user_name}: {similarity_score}")

        # Determine access based on the similarity score and threshold

    # Convert similarity scores to probabilities using softmax
    values = [float(value) for value in users_similarity_scores.values()]
    values_torch = torch.tensor(values, dtype=torch.float32, device=device)
    print(values_torch)
    values_torch = values_torch / torch.min(values_torch)
    print(values_torch)
    users_probabilities = torch.nn.functional.sigmoid(values_torch)
    users_probabilities /= (torch.max(users_probabilities) + .02)
    print(users_probabilities)

    # Step 4: Grant access to authorized users based on threshold
    for granted_user in granted_users:
        user_score = users_similarity_scores[granted_user]
        if user_score >= threshold:
            print("Access granted")
            outputLabel.setText(f"Sentence: {sentence} - Access granted, welcome back Mr. {granted_user}")
            return True, users_probabilities.cpu().numpy(), sentence, sentences_probabilities.cpu().numpy()

    print("Access denied")
    outputLabel.setText(f"Sentence: {sentence} - Acces denied, how else can I help you?")
    return False, users_probabilities.cpu().numpy(), sentence, sentences_probabilities.cpu().numpy()

