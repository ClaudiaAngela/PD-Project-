import torchaudio.transforms as T

def get_mel_transform(sample_rate=16000, n_fft=1024, hop_length=512, n_mels=64):
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )