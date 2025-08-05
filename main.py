import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pandas as pd
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

warnings.filterwarnings("ignore")
import time
start = time.time()

# === CONFIG ===
HC_CSV = r"data/neurovoz/metadata/metadata_hc.csv"
PD_CSV = r"data/neurovoz/metadata/metadata_pd.csv"
PC_MONOLOGUE_CSV = r"data/neurovoz/PC-GITA_16kHz/PCGITA_metadata_monologue.csv"
AUDIO_FOLDER = r"data/neurovoz/audios"
os.makedirs("outputs/mfcc", exist_ok=True)
os.makedirs("outputs/mel", exist_ok=True)
os.makedirs("outputs/waveforms", exist_ok=True)
BATCH_SIZE = 4
EPOCHS_HEAD = 10
EPOCHS_FULL = 10
LR_HEAD = 1e-3
LR_FULL = 1e-4
MAX_LEN = 160
NUM_CLASSES = 2
LABEL_MAP = {0: "HC", 1: "PD"}
NOTEBOOK_FOLDER = "notebook_models"
os.makedirs(NOTEBOOK_FOLDER, exist_ok=True)

# === Custom Dataset ===
class NeurovozDataset(Dataset):
    def __init__(self, df, transform=None, sample_rate=16000, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.sample_rate = sample_rate
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]["audio_path"]
        label = self.df.iloc[idx]["label"]
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if self.transform:
            features = self.transform(waveform)
        else:
            features = waveform
        if features.dim() == 2:
            features = features.unsqueeze(0)
        T = features.shape[-1]
        if T < self.max_len:
            pad = self.max_len - T
            features = torch.nn.functional.pad(features, (0, pad))
        elif T > self.max_len:
            features = features[..., :self.max_len]
        return features, label

# === DataFrame preparation ===
def prepare_neurovoz_dataframe(hc_csv, pd_csv, pc_csv):
    hc_df = pd.read_csv(hc_csv)
    pd_df = pd.read_csv(pd_csv)
    pc_df = pd.read_csv(pc_csv)

    hc_df["label"] = 0
    pd_df["label"] = 1
    hc_df = hc_df[["Audio", "label"]].rename(columns={"Audio": "audio_path"})
    pd_df = pd_df[["Audio", "label"]].rename(columns={"Audio": "audio_path"})
    hc_df["audio_path"] = hc_df["audio_path"].apply(
        lambda x: os.path.normpath(os.path.join(AUDIO_FOLDER, os.path.basename(x))))
    pd_df["audio_path"] = pd_df["audio_path"].apply(
        lambda x: os.path.normpath(os.path.join(AUDIO_FOLDER, os.path.basename(x))))
    # PC-GITA monologue already has absolute path and correct label
    pc_df = pc_df[["audio_path", "label"]]
    df = pd.concat([hc_df, pd_df, pc_df], ignore_index=True)
    print("First 5 absolute paths generated:")
    for path in df["audio_path"].head(5):
        print(f"{path} -- EXISTS: {os.path.isfile(path)}")
    df = df[df["audio_path"].apply(os.path.isfile)].reset_index(drop=True)
    print(f"Total number of examples after filtering: {len(df)}")
    if len(df) == 0:
        print("NO VALID AUDIO FILES EXIST! Check the path and existence of the files.")
    return df

df = prepare_neurovoz_dataframe(HC_CSV, PD_CSV, PC_MONOLOGUE_CSV)
labels_dict = ["HC", "PD"]

# Save ONLY 5 sample images per class (waveform, Mel, MFCC)
def save_sample_images(df, num_per_class=5):
    saved_per_class = {0: 0, 1: 0}
    for i in range(len(df)):
        label = df.iloc[i]["label"]
        if saved_per_class[label] >= num_per_class:
            continue
        audio_path = df.iloc[i]["audio_path"]
        basename = os.path.splitext(os.path.basename(audio_path))[0]
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)

        # Waveform
        plt.figure(figsize=(8, 2))
        plt.plot(waveform.t().numpy())
        plt.title(f"Waveform of {basename} - Label {LABEL_MAP[label]}")
        plt.axis("off")
        plt.savefig(f"outputs/waveforms/waveform_{basename}.png", bbox_inches="tight", pad_inches=0)
        plt.close()

        # MelSpectrogram
        mel = torchaudio.transforms.MelSpectrogram()(waveform)
        plt.figure(figsize=(6, 3))
        plt.imshow(mel[0].numpy(), aspect='auto', origin='lower')
        plt.title(f"MelSpectrogram of {basename} - Label {LABEL_MAP[label]}")
        plt.axis('off')
        plt.savefig(f"outputs/mel/mel_{basename}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        # MFCC
        mfcc = torchaudio.transforms.MFCC()(waveform)
        plt.figure(figsize=(6, 3))
        plt.imshow(mfcc[0].numpy(), aspect='auto', origin='lower')
        plt.title(f"MFCC of {basename} - Label {LABEL_MAP[label]}")
        plt.axis('off')
        plt.savefig(f"outputs/mfcc/mfcc_{basename}.png", bbox_inches='tight', pad_inches=0)
        plt.close()

        saved_per_class[label] += 1
        print(f"Saved waveform, MelSpectrogram and MFCC for {basename}.")
        if all(saved_per_class[l] >= num_per_class for l in LABEL_MAP.keys()):
            break

save_sample_images(df, num_per_class=5)

# === Select feature type (0 = raw, 1 = MFCC, 2 = MelSpectrogram)
data_transform = 2
if data_transform == 1:
    print("MFCC Features classification")
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MFCC(log_mels=False)
    )
elif data_transform == 2:
    print("Mel Spectrogram Features classification")
    train_audio_transforms = nn.Sequential(
        torchaudio.transforms.MelSpectrogram()
    )
else:
    train_audio_transforms = None

# === Stratified train/test split ===
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=42
)

# === Create dataset/dataLoader ===
train_dataset = NeurovozDataset(train_df, transform=train_audio_transforms)
test_dataset = NeurovozDataset(test_df, transform=train_audio_transforms)
trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Pretrained ResNet18, adapted for 1-channel input and 2 classes ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights="IMAGENET1K_V1")
model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# === Train/test functions + metrics ===
def train(model, loader, optimizer, scheduler, criterion, epoch, device):
    model.train()
    total_loss = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(loader.dataset)
    print(f"[Epoch {epoch+1}] Train loss: {avg_loss:.4f}")
    return avg_loss

def test(model, loader, criterion, device):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            pred = out.argmax(1)
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())
            correct += (pred == yb).sum().item()
            total += yb.size(0)
    acc = correct / total if total else 0
    print(f"Test accuracy: {acc:.3f}")
    return acc, all_labels, all_preds

train_losses = []
test_accuracies = []

# === Fine-tuning: first only last layer, then all layers ===
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR_HEAD)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR_HEAD,
    steps_per_epoch=len(trainloader),
    epochs=EPOCHS_HEAD,
    anneal_strategy='linear'
)
criterion = nn.CrossEntropyLoss()

print("\n=== FINE-TUNING only the head (last layer) ===")
for epoch in range(EPOCHS_HEAD):
    loss = train(model, trainloader, optimizer, scheduler, criterion, epoch, device)
    train_losses.append(loss)
    acc, _, _ = test(model, testloader, criterion, device)
    test_accuracies.append(acc)

# === Fine-tuning ALL layers ===
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=LR_FULL)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR_FULL,
    steps_per_epoch=len(trainloader),
    epochs=EPOCHS_FULL,
    anneal_strategy='linear'
)

print("\n=== FINE-TUNING ALL layers ===")
for epoch in range(EPOCHS_FULL):
    loss = train(model, trainloader, optimizer, scheduler, criterion, epoch, device)
    train_losses.append(loss)
    acc, _, _ = test(model, testloader, criterion, device)
    test_accuracies.append(acc)

# === Final evaluation: Confusion matrix and F1-score on test set ===
print("\n=== Final evaluation on test set ===")
_, all_labels, all_preds = test(model, testloader, criterion, device)
cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HC", "PD"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("outputs/confusion_matrix.png")
plt.close()

report = classification_report(all_labels, all_preds, target_names=["HC", "PD"])
with open("outputs/classification_report.txt", "w") as f:
    f.write(report)
print("Classification report:\n", report)

# === Accuracy graph over epochs ===
plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, len(test_accuracies) + 1), test_accuracies, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Test accuracy")
plt.title("Test accuracy over epochs")
plt.grid()
plt.savefig("outputs/test_accuracy_epochs.png")
plt.close()
print(f"Total time: {(time.time()-start)/60:.2f} minutes")
# === Save the model in the notebook folder ===
torch.save(model.state_dict(), os.path.join(NOTEBOOK_FOLDER, "model_neurovoz_resnet18_finetuned.pth"))
print(f"Model saved as {os.path.join(NOTEBOOK_FOLDER, 'model_neurovoz_resnet18_finetuned.pth')}")
print("Confusion matrix, classification report and accuracy graph have been saved in outputs/")