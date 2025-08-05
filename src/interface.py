import argparse
import torch
import torchaudio
import matplotlib.pyplot as plt
import os
import torchvision.models as models
import torch.nn as nn

# Config
MODEL_PATH = "model_neurovoz_resnet18_finetuned.pth"
NUM_CLASSES = 2
LABEL_MAP = {0: "HC", 1: "PD"}
SAMPLE_RATE = 16000
MAX_LEN = 160

def preprocess_wav(filepath, transform=None, sample_rate=SAMPLE_RATE, max_len=MAX_LEN):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    if transform is not None:
        features = transform(waveform)
    else:
        features = waveform
    if features.dim() == 2:
        features = features.unsqueeze(0)
    T = features.shape[-1]
    if T < max_len:
        pad = max_len - T
        features = torch.nn.functional.pad(features, (0, pad))
    elif T > max_len:
        features = features[..., :max_len]
    return features

def load_model(model_path):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

def predict(filepath, model_path=MODEL_PATH):
    transform = torchaudio.transforms.MelSpectrogram()
    input_tensor = preprocess_wav(filepath, transform=transform)
    input_tensor = input_tensor.unsqueeze(0)  # [batch, 1, 64, 160]
    model = load_model(model_path)
    with torch.no_grad():
        output = model(input_tensor)
        pred_class = output.argmax(dim=1).item()
        confidence = torch.softmax(output, dim=1)[0, pred_class].item()
    print(f"Predicție pentru '{os.path.basename(filepath)}': {LABEL_MAP[pred_class]} (confidență: {confidence:.2f})")
    return pred_class, confidence

def plot_waveform_and_mel(filepath):
    waveform, sr = torchaudio.load(filepath)
    plt.figure(figsize=(10, 3))
    plt.plot(waveform.t().numpy())
    plt.title("Forma de undă audio")
    plt.show()
    mel = torchaudio.transforms.MelSpectrogram()(waveform)
    plt.figure(figsize=(10, 3))
    plt.imshow(mel[0].numpy(), aspect='auto', origin='lower')
    plt.title("MelSpectrogram")
    plt.show()

def main():
    print("Script pornit!")
    parser = argparse.ArgumentParser(description="Inference script for Neurovoz ResNet18 model")
    parser.add_argument("wav_path", help="Path to .wav file")
    parser.add_argument("--model", default=MODEL_PATH, help="Path to model .pth file")
    parser.add_argument("--show", action="store_true", help="Show waveform and MelSpectrogram")
    args = parser.parse_args()
    print(f"Argumente citite: {args}")

    if not os.path.isfile(args.wav_path):
        print(f"Fișierul '{args.wav_path}' nu există!")
        exit(1)

    if not os.path.isfile(args.model):
        print(f"Modelul '{args.model}' nu există!")
        exit(1)

    pred_class, confidence = predict(args.wav_path, args.model)
    if args.show:
        plot_waveform_and_mel(args.wav_path)

if __name__ == "__main__":
    main()