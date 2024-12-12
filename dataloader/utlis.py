import numpy as np
import torch

def get_sub_seqs(x_arr, seq_len=100, stride=1):
    """

    Parameters
    ----------
    x_arr: np.array, required
        input original data with shape [time_length, channels]

    seq_len: int, optional (default=100)
        Size of window used to create subsequences from the data

    stride: int, optional (default=1)
        number of time points the window will move between two subsequences

    Returns
    -------
    x_seqs: np.array
        Split sub-sequences of input time-series data
    """

    if x_arr.shape[0] < seq_len:
        seq_len = x_arr.shape[0]
    seq_starts = np.arange(0, x_arr.shape[0] - seq_len + 1, stride)
    x_seqs = np.array([x_arr[i:i + seq_len] for i in seq_starts])

    return x_seqs


def augment_data(self, spec, imu, audio_tensor):
    
    spec = torch.flip(spec, [1])
    imu = torch.flip(imu, [0])
    audio_tensor = torch.flip(audio_tensor, [0])
    return spec, imu, audio_tensor

def pulse_audio(self,length=4800, pulse_amplitude=10, noise_std=1):
    audio = np.random.normal(0, noise_std, length)
    pulse_position = np.random.randint(0, length)
    pulse = np.random.uniform(pulse_amplitude, pulse_amplitude * 2)
    audio[pulse_position] += pulse
    return audio
def continuous_audio(self,length=4800, noise_std=1, amplification_factor=2):
    audio = np.random.normal(0, noise_std, length)
    audio[length//2:] *= amplification_factor
    return audio

def pulse_imu(self,length=200, pulse_amplitude=10, noise_std=1):
    imu = np.random.normal(0, noise_std, length)
    pulse_position = np.random.randint(0, length)
    pulse = np.random.uniform(pulse_amplitude, pulse_amplitude * 2)
    imu[pulse_position] += pulse
    return imu
def continuous_imu(self,length=200, noise_std=1, amplification_factor=2):

    imu = np.random.normal(0, noise_std, length)
    imu[length//2:] *= amplification_factor
    return imu


def apply_mask(imu):
    channels, samples = imu.shape
    mask_ratio = np.random.uniform(0.4, 0.6)
    mask_length = int(samples * mask_ratio)
    train_data = imu[:, :mask_length]
    test_data = imu[:, mask_length:]

    def interpolate_data(data, original_length):
        num_channels, data_length = data.shape
        new_indices = np.linspace(0, data_length - 1, original_length)

        interpolated_data = np.zeros((num_channels, original_length))
        for i in range(num_channels):
            interpolated_data[i] = np.interp(new_indices, np.arange(data_length), data[i])

        return interpolated_data

    train_data_stretched = interpolate_data(train_data, samples)
    test_data_stretched = interpolate_data(test_data, samples)

    return train_data_stretched, test_data_stretched
