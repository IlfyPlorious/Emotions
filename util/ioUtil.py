import os
from os.path import join

import librosa
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from IPython.display import Audio, display
from torchvision.io import read_image

from util import AudioFileModel

parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
crema_d_dir = os.path.join(parent_dir, 'CREMA-D')
audio_wav_dir = os.path.join(crema_d_dir, 'AudioWAV')
labels = {
    'ANGER': 0,
    'DISGUST': 1,
    'FEAR': 2,
    'HAPPY': 3,
    'NEUTRAL': 4,
    'SAD': 5
}


def get_wav_files(limit=4000):
    wav_files = os.listdir(audio_wav_dir)
    audio_files_list = list()
    for file in wav_files[:limit]:
        actor, sample, emotion, emotion_level = file.split('_')
        emotion = get_emotion_by_notation(emotion)
        emotion_level = get_emotion_level_by_notation(emotion_level)
        wav_file_path = join(audio_wav_dir, file)
        metadata = torchaudio.info(wav_file_path)
        # _, waveform_data = wavfile.read(wav_file_path)
        waveform_data, sample_rate = torchaudio.load(wav_file_path)
        audio_file = AudioFileModel.AudioFile(sample=sample, actor=actor, emotion=emotion,
                                              emotion_level=emotion_level, metadata=metadata,
                                              waveform_data=waveform_data, sample_rate=sample_rate)
        if audio_file.get_length_in_seconds() < 3:
            audio_files_list.append(audio_file)

    return audio_files_list


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_specgram(file, title="Spectrogram", save_dir=None, xlim=None):
    waveform = file.waveform_data.numpy()
    sample_rate = file.sample_rate
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c + 1}')
        if xlim:
            axes[c].set_xlim(xlim)
    if save_dir:
        plt.axis('off')
        file_path = os.path.join(save_dir, file.get_file_name())
        plt.savefig(file_path, bbox_inches='tight', transparent=True, pad_inches=0)
        plt.close()
    else:
        figure.suptitle(title)
        plt.show()


def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    if num_channels == 1:
        display(Audio(waveform[0], rate=sample_rate))
    elif num_channels == 2:
        display(Audio((waveform[0], waveform[1]), rate=sample_rate))
    else:
        raise ValueError("Waveform with more than 2 channels are not supported.")


def read_image_from_file(img_dir, file):
    img_path = os.path.join(img_dir, file.get_file_name())
    label = file.get_file_name().split('_')[2]
    image = read_image(img_path)

    # image is initially [channels, width, height], but plt.imshow() needs [width, height, channels]
    # image = torch.permute(image, [1, 2, 0])

    return image, label


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(spec, origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show()


def get_spectrogram_from_waveform_in_db(waveform, stretched=False):
    n_fft = 1024
    win_length = None
    hop_length = 512

    # define transformation
    spectrogram = T.Spectrogram(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
    )
    # Perform transformation
    if stretched:
        final_spectrogram = stretch_spectrogram(spectrogram(waveform), n_freq=n_fft // 2 + 1, hop_length=hop_length)
    else:
        final_spectrogram = spectrogram(waveform)

    return librosa.power_to_db(final_spectrogram)

    # Spectrogram size will be 513 on x_axis


def stretch_spectrogram(spectrogram, index=1, final_dim=100, n_freq=201, hop_length=None):
    stretch = T.TimeStretch(n_freq=n_freq, hop_length=hop_length)
    rate = len(spectrogram[0][index]) / final_dim
    stretched_spectrogram = stretch(spectrogram, rate)
    if stretched_spectrogram.shape[2] != 100:
        rate = len(stretched_spectrogram[0][index]) / final_dim
        stretched_spectrogram = stretch(stretched_spectrogram, rate)

    return stretched_spectrogram


def save_spectrogram_data(file, save_dir='SpectrogramData'):
    spectrogram = get_spectrogram_from_waveform_in_db(file.waveform_data, stretched=True)
    actor_dir = os.path.join(save_dir, file.actor)
    saved_file_name = os.path.join(actor_dir, file.get_file_name())

    if not os.path.exists(actor_dir):
        os.makedirs(actor_dir)

    np.save(saved_file_name, spectrogram)


def get_waveform_from_spectrogram(spectrogram):
    n_fft = 1024
    win_length = None
    hop_length = 512

    griffin_lim = T.GriffinLim(
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    # Transform into waveform
    return griffin_lim(spectrogram)


def get_stretched_spectrogram(spectrogram, rate=1):
    stretch = torchaudio.transforms.TimeStretch(n_freq=1)
    return stretch(spectrogram, rate)


def get_pitch_from_waveform(waveform, sample_rate):
    return F.detect_pitch_frequency(waveform, sample_rate)


def get_emotion_level_by_notation(notation):
    if notation == 'LO':
        return "LOW"
    elif notation == 'MD':
        return "MEDIUM"
    elif notation == 'HI':
        return "HIGH"
    else:
        return "UNSPECIFIED"


def get_emotion_by_notation(notation):
    if notation == 'ANG':
        return "ANGER"
    elif notation == 'DIS':
        return "DISGUST"
    elif notation == 'FEA':
        return "FEAR"
    elif notation == 'HAP':
        return "HAPPY"
    elif notation == 'SAD':
        return "SAD"
    else:
        return "NEUTRAL"


def save_spectrograms_to_dir(spectrograms_count=500, dir_name='Spectrograms'):
    for file in get_wav_files(spectrograms_count):
        save_dir = os.path.join(parent_dir, dir_name)
        file_name = f"{file.actor}_{file.sample}_{file.emotion}_{file.emotion_level}"
        print(f'Saving {file_name}...')
        plot_specgram(waveform=file.waveform_data, sample_rate=file.sample_rate, save_dir=save_dir,
                      file_name=file_name)
        print(f'File {file_name} saved in {save_dir}')
