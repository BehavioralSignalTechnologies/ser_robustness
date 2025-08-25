import numpy as np
import matplotlib.pyplot as plt
import librosa.display
import librosa, os
import soundfile as sf
import os

def get_spectrogram(audio_path, label):
    audio_data, sample_rate = librosa.load(audio_path, sr=None)
    spectrogram = np.abs(librosa.stft(audio_data))

    # Display spectrogram
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=sample_rate, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(label)
    plt.show()


# Gaussian Noise
from robuser.corruptions.gaussian import AWGNAugmentation

def gaussian_corruption(label, audio_data, sample_rate):
    output_file_path = f"{label}_gaussian_10.wav"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    gaussian_config = {
        'snr': 10,
    }

    gaussian_10 = AWGNAugmentation(gaussian_config)
    corrupted_audio, _ = gaussian_10.run(audio_data, sample_rate)

    sf.write(output_file_path, corrupted_audio, sample_rate)
    
    return output_file_path


# Clipping distortion
from robuser.corruptions.clipping_distortion import AddClippingDistortion

def clipping_corruption(label, audio_data, sample_rate):
    output_file_path = f"{label}_clipping_40.wav"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    clip_config = {
        'max_percentile_threshold': 40,
    }

    clipping_40 = AddClippingDistortion(clip_config)
    corrupted_audio, _ = clipping_40.run(audio_data, sample_rate)

    sf.write(output_file_path, corrupted_audio, sample_rate)

    return output_file_path

# Gain Transition
from robuser.corruptions.gain_transition import AddGainTransition

def gain_corruption(label, audio_data, sample_rate):
    output_file_path = f"{label}_gain_transition_30_10.wav"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    gain_config = {
        'min_max_gain_db': [-30.0, -10.0]
    }

    gain_transition_30_10 = AddGainTransition(gain_config)
    corrupted_audio, _ = gain_transition_30_10.run(audio_data, sample_rate)

    sf.write(output_file_path, corrupted_audio, sample_rate)
    
    return output_file_path


# Compression artifacts
from robuser.corruptions.compression import Compression

def compress_audio(audio_file_path, label, sample_rate):
    output_file_path = f"{label}_compressed_8kbps.wav"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    compress_config = {
        'bit_rate': 8,
    }

    compression_8 = Compression(compress_config)
    corrupted_audio, _ = compression_8.run(audio_file_path, sample_rate)

    sf.write(output_file_path, corrupted_audio, sample_rate)

    return output_file_path


# Reverberation
from robuser.corruptions.impulse_response import AddImpulseResponse
import warnings

def reverberation(label, audio_data, sample_rate):
    warnings.filterwarnings("ignore")
    output_file_path = f"{label}_reverb_underground_01_05.wav"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    reverb_config = {
        'ir_path': "../datasets/EchoThiefImpulseResponseLibrary/Underground",
        'rt60_range': [0.1, 0.5],
    }

    reverberation_01_05 = AddImpulseResponse(reverb_config)
    corrupted_audio, _ = reverberation_01_05.run(audio_data, sample_rate)

    sf.write(output_file_path, corrupted_audio, sample_rate)
    
    return output_file_path


# Add background noise
esc_config = {
    'content_dataset_path': '../datasets/ESC-50-master',
    'snr': 0
}

musan_config = {
    'content_dataset_path': '../datasets/musan',
    'snr': 10
}

urban_config = {
    'content_dataset_path': '../datasets/urbansound8k',
    'snr': 20
}

from robuser.corruptions.content import ContentCorruption

def background_noise(label, audio_data, sample_rate, config):
    noise_dataset = os.path.basename(config['content_dataset_path'])
    snr_level = str(config['snr'])
    output_file_path = f"{label}_{noise_dataset}_{snr_level}_db.wav"
    if os.path.exists(output_file_path):
        os.remove(output_file_path)

    esc_augment_0_db = ContentCorruption(config)
    corrupted_audio, corruption_type = esc_augment_0_db.run(audio_data, sample_rate)

    print(f"Corruption file: {corruption_type}")
    sf.write(output_file_path, corrupted_audio, sample_rate)
    
    return output_file_path
