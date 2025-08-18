import os
import random
import torch
import torchaudio
import torchvision
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# --- Configs ---
SR = 16000
SEGMENT_SEC = 3.0
SEGMENT_LENGTH = int(SR * SEGMENT_SEC)
OVERLAP_SEC = 1.0
OVERLAP = int(SR * OVERLAP_SEC)
BATCH_SIZE = 16
EPOCHS = 30
N_MELS = 224
N_FRAMES = 224
N_MFCC = 13
MODELS = ['resnet18']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 2

def segment_audio(waveform, segment_length, overlap):
    # Segment the waveform into overlapping segments
    segments = []
    step = segment_length - overlap
    for start in range(0, waveform.shape[-1] - segment_length + 1, step):
        seg = waveform[..., start:start + segment_length]
        segments.append(seg)
    return segments

def save_spec_image(spec, filename):
    # Save a spectrogram image (normalized)
    spec_np = spec.permute(1,2,0).cpu().numpy()
    spec_np = (spec_np - spec_np.min()) / (spec_np.max() - spec_np.min() + 1e-8)
    plt.imsave(filename, spec_np)

def save_mfcc(mfcc, filename):
    # Save MFCC features to a .npy file
    mfcc_np = mfcc.cpu().numpy()
    np.save(filename, mfcc_np)

def save_waveform(waveform, filename):
    # Save waveform to a .npy file
    waveform_np = waveform.cpu().numpy()
    np.save(filename, waveform_np)

def waveform_to_spec_and_mfcc(waveform):
    # Convert waveform to mel spectrogram and MFCC features
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_fft=1024,
        hop_length=int((waveform.shape[-1]-1)/(N_FRAMES-1)),
        n_mels=N_MELS
    )
    mel = mel_transform(waveform)  # [1, n_mels, n_frames]
    mel = torchaudio.transforms.AmplitudeToDB()(mel)
    if mel.dim() == 3 and mel.shape[0] == 1:
        mel = mel.squeeze(0)
    if mel.shape[1] > N_FRAMES:
        mel = mel[:, :N_FRAMES]
    elif mel.shape[1] < N_FRAMES:
        mel = torch.nn.functional.pad(mel, (0, N_FRAMES - mel.shape[1]))
    mel_rgb = mel.unsqueeze(0).repeat(3, 1, 1) # [3, N_MELS, N_FRAMES]

    mfcc_transform = torchaudio.transforms.MFCC(
        sample_rate=SR,
        n_mfcc=N_MFCC,
        melkwargs={
            "n_fft": 1024,
            "n_mels": N_MELS,
            "hop_length": int((waveform.shape[-1]-1)/(N_FRAMES-1)),
            "mel_scale": "htk"
        }
    )
    mfcc = mfcc_transform(waveform)  # [1, n_mfcc, n_frames]
    if mfcc.dim() == 3 and mfcc.shape[0] == 1:
        mfcc = mfcc.squeeze(0)
    if mfcc.shape[1] > N_FRAMES:
        mfcc = mfcc[:, :N_FRAMES]
    elif mfcc.shape[1] < N_FRAMES:
        mfcc = torch.nn.functional.pad(mfcc, (0, N_FRAMES - mfcc.shape[1]))
    return mel_rgb, mfcc

class SpeechDataset(Dataset):
    def __init__(self, file_df, augment=False, save_examples=False, save_limit=5):
        self.file_df = file_df.copy()
        self.augment = augment
        self.save_examples = save_examples
        self.save_limit = save_limit
        self.examples_saved = 0

        # Create output folders for waveform, MFCC, and mel spectrogram
        self.wave_dir = "outputs/waveform"
        self.mfcc_dir = "outputs/mfcc"
        self.mel_dir = "outputs/melspec"
        if save_examples:
            os.makedirs(self.wave_dir, exist_ok=True)
            os.makedirs(self.mfcc_dir, exist_ok=True)
            os.makedirs(self.mel_dir, exist_ok=True)

    def __len__(self):
        return len(self.file_df)

    def __getitem__(self, idx):
        row = self.file_df.iloc[idx]
        wav, sr = torchaudio.load(row['filepath'])
        # If the waveform is longer than the segment length, randomly select a segment
        if wav.shape[1] > SEGMENT_LENGTH:
            segments = segment_audio(wav, SEGMENT_LENGTH, OVERLAP)
            wav = random.choice(segments)
        else:
            # If the waveform is shorter, pad it
            if wav.shape[1] < SEGMENT_LENGTH:
                wav = torch.nn.functional.pad(wav, (0, SEGMENT_LENGTH - wav.shape[1]))
        # If augmentation is enabled, add noise
        if self.augment:
            wav = wav + 0.01 * torch.randn_like(wav)
        spec, mfcc = waveform_to_spec_and_mfcc(wav)
        # Save only for the first save_limit examples
        if self.save_examples and self.examples_saved < self.save_limit:
            base_name = f"{idx}_{row['label']}_{row['corpus']}"
            save_waveform(wav, os.path.join(self.wave_dir, f"{base_name}.npy"))
            save_mfcc(mfcc, os.path.join(self.mfcc_dir, f"{base_name}.npy"))
            save_spec_image(spec, os.path.join(self.mel_dir, f"{base_name}.jpg"))
            self.examples_saved += 1
        label = 0 if row['label'] == 'HC' else 1
        return spec, label, mfcc

def prepare_dataframe(neurovoz_dir, pcgita_monologue_dir, sentences_dir):
    data_rows = []
    # Neurovoz: label is taken directly from the file name
    for root, _, files in os.walk(neurovoz_dir):
        for f in files:
            if f.endswith('.wav'):
                label = f.split('_')[0]  # 'PD' or 'HC' at the beginning
                subject = f.split('_')[1] if '_' in f else f.split('.')[0]
                if label not in ['PD', 'HC']:  # Skip if the label is not correct
                    continue
                data_rows.append({'filepath': os.path.join(root, f), 'label': label, 'corpus': 'neurovoz', 'subject': subject})
    # PC-GITA monologue: folders PD/HC
    for label in ['PD', 'HC']:
        label_dir = os.path.join(pcgita_monologue_dir, label)
        if not os.path.exists(label_dir):
            continue
        for root, _, files in os.walk(label_dir):
            for f in files:
                if f.endswith('.wav'):
                    subject = f.split('_')[0]  # or another logic, depending on your format
                    data_rows.append({'filepath': os.path.join(root, f), 'label': label, 'corpus': 'pcgita_monologue', 'subject': subject})
    # Sentences: same as neurovoz, if you use it
    for root, _, files in os.walk(sentences_dir):
        for f in files:
            if f.endswith('.wav'):
                label = f.split('_')[0]
                subject = f.split('_')[1] if '_' in f else f.split('.')[0]
                if label not in ['PD', 'HC']:
                    continue
                data_rows.append({'filepath': os.path.join(root, f), 'label': label, 'corpus': 'sentence', 'subject': subject})
    df = pd.DataFrame(data_rows)
    return df

def stratified_group_split(df):
    # Stratified split by subject and label
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    X = df['filepath']
    y = df['label']
    groups = df['subject']
    for train_idx, test_idx in sgkf.split(X, y, groups):
        train_labels = set(df.iloc[train_idx]['label'].unique())
        test_labels = set(df.iloc[test_idx]['label'].unique())
        # Ensure both train and test have both labels
        if train_labels == {'HC', 'PD'} and test_labels == {'HC', 'PD'}:
            return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    # Fallback: split by file if valid subject split is not found
    print("WARNING: Subject split failed. Using file-level stratified split!")
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def get_model(model_name):
    if model_name == 'resnet18':
        # Create a ResNet18 model with 3-channel input and correct output classes
        model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    else:
        raise ValueError("Unknown model name")
    return model

def train_model(model, train_dl, test_dl, epochs=EPOCHS, lr=1e-4, model_name="model"):
    # Train the model and evaluate on test data
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_acc = 0
    best_state = None
    train_losses, test_accs = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for batch in train_dl:
            specs, labels, mfccs = batch
            specs, labels = specs.to(DEVICE, dtype=torch.float), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_dl))
        # Evaluate
        model.eval()
        correct, total = 0, 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in test_dl:
                specs, labels, mfccs = batch
                specs, labels = specs.to(DEVICE, dtype=torch.float), labels.to(DEVICE)
                outputs = model(specs)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total > 0 else 0
        test_accs.append(acc)
        if acc > best_acc:
            best_acc = acc
            best_state = model.state_dict()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_losses[-1]:.4f}, Test Acc: {acc:.4f}")
    # Save results directly in outputs/results (no subfolder)
    os.makedirs("outputs/results", exist_ok=True)
    # Save models in notebook_models
    os.makedirs("notebook_models", exist_ok=True)
    torch.save(best_state, f"notebook_models/best_model_{model_name}.pt")
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), test_accs, label='Test Acc')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(f"outputs/results/learning_curve_{model_name}.png")
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure()
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix')
    # Add numbers to the squares
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black', fontsize=16)
    plt.savefig(f"outputs/results/confusion_matrix_{model_name}.png")
    with open(f"outputs/results/classification_report_{model_name}.txt", "w") as f:
        f.write(classification_report(all_labels, all_preds))

def run_verifications(train_df, test_df, train_dl, test_dl):
    print("\n--- Train/Test label verifications ---")
    print("Train distribution:\n", train_df['label'].value_counts())
    print("Test distribution:\n", test_df['label'].value_counts())
    print("\n--- A batch from train_dl (labels) ---")
    for specs, labels, mfccs in train_dl:
        print("Train batch labels:", labels)
        print("Train batch specs shape:", specs.shape)
        print("Train batch mfcc shape:", mfccs.shape)
        break
    print("\n--- A batch from test_dl (labels) ---")
    for specs, labels, mfccs in test_dl:
        print("Test batch labels:", labels)
        print("Test batch specs shape:", specs.shape)
        print("Test batch mfcc shape:", mfccs.shape)
        break

def main():
    os.makedirs("outputs/results", exist_ok=True)
    # Set directories (modify these to match your structure)
    neurovoz_dir = "data/neurovoz"
    pcgita_monologue_dir = "data/pcgita/monologue"
    sentences_dir = "data/sentences"
    # Build the dataframe
    df = prepare_dataframe(neurovoz_dir, pcgita_monologue_dir, sentences_dir)
    # Debug subject and label distribution
    print("Subjects per label:")
    print(df.groupby('label')['subject'].nunique())
    print("Label counts:")
    print(df['label'].value_counts())
    # Split train/test
    train_df, test_df = stratified_group_split(df)
    # Save 5 examples in waveform/mfcc/melspec (no train/test subfolders)
    all_df = pd.concat([train_df, test_df]).reset_index(drop=True)
    ds = SpeechDataset(all_df, augment=False, save_examples=True, save_limit=5)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for _ in dl:
        pass
    # Datasets for training
    train_ds = SpeechDataset(train_df, augment=True, save_examples=False)
    test_ds = SpeechDataset(test_df, augment=False, save_examples=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    # Useful verifications
    run_verifications(train_df, test_df, train_dl, test_dl)
    for model_name in MODELS:
        print(f"\nTraining model: {model_name}")
        model = get_model(model_name)
        train_model(model, train_dl, test_dl, epochs=EPOCHS, model_name=model_name)
        print(f"Finished training {model_name}")

if __name__ == "__main__":
    main()