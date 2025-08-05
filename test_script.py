import os
import soundfile as sf

path = r'C:\Users\claud\PycharmProjects\pytorch_project\pytorch_project\data\speechcommands\SpeechCommands\backward\0165e0e8_nohash_0.wav'
print("OS exists:", os.path.exists(path))

try:
    data, samplerate = sf.read(path)
    print("Soundfile read OK:", data.shape, samplerate)
except Exception as e:
    print("Soundfile error:", e)

try:
    import torchaudio
    waveform, sample_rate = torchaudio.load(path)
    print("Torchaudio read OK:", waveform.shape, sample_rate)
except Exception as e:
    print("Torchaudio error:", e)