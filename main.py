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
import random
import time

warnings.filterwarnings("ignore")
start = time.time()

# === CONFIG ===
NEUROVOZ_HC_CSV = r"data/neurovoz/metadata/metadata_hc.csv"
NEUROVOZ_PD_CSV = r"data/neurovoz/metadata/metadata_pd.csv"
NEUROVOZ_AUDIO_FOLDER = r"data/neurovoz/audios"
PCGITA_MONO_HC = r"C:\Users\claud\PycharmProjects\pytorch_project\pytorch_project\data\PC-GITA_16kHz\PC-GITA_16kHz\monologue\sin normalizar\hc"
PCGITA_MONO_PD = r"C:\Users\claud\PycharmProjects\pytorch_project\pytorch_project\data\PC-GITA_16kHz\PC-GITA_16kHz\monologue\sin normalizar\pd"
os.makedirs("outputs/mfcc", exist_ok=True)
os.makedirs("outputs/mel", exist_ok=True)
os.makedirs("outputs/waveforms", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)
BATCH_SIZE = 4
EPOCHS_HEAD = 2
EPOCHS_TAIL = 3
LR_HEAD = 2e-4
LR_TAIL = 8e-5
MAX_LEN = 160
NUM_CLASSES = 2
LABEL_MAP = {0: "HC", 1: "PD"}

# === GRIDSEARCH parameter combinations ===
param_combos = [
    {"n_fft": 512, "hop_length": 160, "win_length": 512, "n_mels": 64},
    {"n_fft": 400, "hop_length": 160, "win_length": 400, "n_mels": 64},
    {"n_fft": 256, "hop_length": 80, "win_length": 256, "n_mels": 64},
]

# === LIGHT AUDIO AUGMENTATION ===
def augment_waveform(waveform, sr):
    if random.random() < 0.3:
        waveform = waveform + 0.003 * torch.randn_like(waveform)
    if random.random() < 0.3:
        gain = random.uniform(0.9, 1.1)
        waveform = waveform * gain
    orig_len = waveform.shape[1]
    crop_len = int(orig_len * random.uniform(0.9, 1.0))
    if random.random() < 0.2 and crop_len < orig_len:
        start = random.randint(0, orig_len - crop_len)
        waveform = waveform[:, start:start+crop_len]
        if waveform.shape[1] < orig_len:
            waveform = torch.nn.functional.pad(waveform, (0, orig_len - waveform.shape[1]))
    return waveform

# === DUAL INPUT DATASET: Mel + MFCC as 2 channels ===
class DualInputDataset(Dataset):
    def __init__(self, df, mel_params, mfcc_params, sample_rate=16000, max_len=MAX_LEN, train=False, augment=False):
        self.df = df.reset_index(drop=True)
        self.sample_rate = sample_rate
        self.max_len = max_len
        self.train = train
        self.augment = augment
        self.mel_transform = torchaudio.transforms.MelSpectrogram(**mel_params)
        self.mfcc_transform = torchaudio.transforms.MFCC(**mfcc_params)
        self.n_mels = mel_params["n_mels"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.df.iloc[idx]["audio_path"]
        label = self.df.iloc[idx]["label"]
        waveform, sr = torchaudio.load(audio_path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if self.augment and self.train:
            waveform = augment_waveform(waveform, self.sample_rate)
        waveform = waveform - waveform.mean()
        waveform = waveform / (waveform.std() + 1e-7)
        mel = self.mel_transform(waveform)
        mfcc = self.mfcc_transform(waveform)
        T = min(mel.shape[-1], mfcc.shape[-1], self.max_len)
        mel = mel[..., :T]
        mfcc = mfcc[..., :T]
        if mfcc.shape[1] < self.n_mels:
            pad_mfcc = torch.zeros(1, self.n_mels - mfcc.shape[1], T)
            mfcc = torch.cat([mfcc, pad_mfcc], dim=1)
        mel = mel[:, :self.n_mels, :]
        mfcc = mfcc[:, :self.n_mels, :]
        features = torch.cat([mel, mfcc], dim=0)
        if features.shape[-1] < self.max_len:
            pad = self.max_len - features.shape[-1]
            features = torch.nn.functional.pad(features, (0, pad))
        return features, label

# === READ NEUROVOZ CSV ===
def prepare_neurovoz_df(hc_csv, pd_csv, folder):
    hc_df = pd.read_csv(hc_csv)
    pd_df = pd.read_csv(pd_csv)
    hc_df["label"] = 0
    pd_df["label"] = 1
    hc_df = hc_df[["Audio", "label"]].rename(columns={"Audio": "audio_path"})
    pd_df = pd_df[["Audio", "label"]].rename(columns={"Audio": "audio_path"})
    hc_df["audio_path"] = hc_df["audio_path"].apply(
        lambda x: os.path.normpath(os.path.join(folder, os.path.basename(x))))
    pd_df["audio_path"] = pd_df["audio_path"].apply(
        lambda x: os.path.normpath(os.path.join(folder, os.path.basename(x))))
    df = pd.concat([hc_df, pd_df], ignore_index=True)
    df = df[df["audio_path"].apply(os.path.isfile)].reset_index(drop=True)
    df["corpus"] = "neurovoz"
    return df

# === READ PC-GITA MONOLOGUE ===
def prepare_pcgita_monologue_df(hc_folder, pd_folder):
    hc_files = [os.path.join(hc_folder, f) for f in os.listdir(hc_folder) if f.lower().endswith(".wav")]
    pd_files = [os.path.join(pd_folder, f) for f in os.listdir(pd_folder) if f.lower().endswith(".wav")]
    hc_df = pd.DataFrame({"audio_path": hc_files, "label": 0})
    pd_df = pd.DataFrame({"audio_path": pd_files, "label": 1})
    df = pd.concat([hc_df, pd_df], ignore_index=True)
    df = df[df["audio_path"].apply(os.path.isfile)].reset_index(drop=True)
    df["corpus"] = "pcgita"
    return df

# === Combine data ===
neurovoz_df = prepare_neurovoz_df(NEUROVOZ_HC_CSV, NEUROVOZ_PD_CSV, NEUROVOZ_AUDIO_FOLDER)
pcgita_df = prepare_pcgita_monologue_df(PCGITA_MONO_HC, PCGITA_MONO_PD)
full_df = pd.concat([neurovoz_df, pcgita_df], ignore_index=True)
print(f"Total samples: {len(full_df)} | Neurovoz: {len(neurovoz_df)} | PC-GITA: {len(pcgita_df)}")

# === Save 2 examples per corpus/label/param_combo ===
def save_sample_images(df, num_per_class=2, mel_params=None, mfcc_params=None, tag=""):
    for corpus in ["neurovoz", "pcgita"]:
        for label in [0, 1]:
            saved = 0
            subdf = df[(df["corpus"]==corpus) & (df["label"]==label)]
            for i in range(len(subdf)):
                audio_path = subdf.iloc[i]["audio_path"]
                basename = os.path.basename(audio_path).replace('.wav','')
                waveform, sr = torchaudio.load(audio_path)
                if sr != 16000:
                    waveform = torchaudio.functional.resample(waveform, sr, 16000)
                # Waveform
                plt.figure(figsize=(8,2))
                plt.plot(waveform.t().numpy())
                plt.title(f"{corpus} - Waveform {basename} - {LABEL_MAP[label]}")
                plt.axis("off")
                plt.savefig(f"outputs/waveforms/waveform_{corpus}_{basename}_{tag}.png", bbox_inches="tight", pad_inches=0)
                plt.close()
                # Mel
                mel = torchaudio.transforms.MelSpectrogram(**mel_params)(waveform)
                plt.figure(figsize=(6,3))
                plt.imshow(mel[0].numpy(), aspect='auto', origin='lower')
                plt.title(f"{corpus} - Mel {basename} - {LABEL_MAP[label]}")
                plt.axis('off')
                plt.savefig(f"outputs/mel/mel_{corpus}_{basename}_{tag}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
                # MFCC
                mfcc = torchaudio.transforms.MFCC(**mfcc_params)(waveform)
                plt.figure(figsize=(6,3))
                plt.imshow(mfcc[0].numpy(), aspect='auto', origin='lower')
                plt.title(f"{corpus} - MFCC {basename} - {LABEL_MAP[label]}")
                plt.axis('off')
                plt.savefig(f"outputs/mfcc/mfcc_{corpus}_{basename}_{tag}.png", bbox_inches='tight', pad_inches=0)
                plt.close()
                saved += 1
                if saved >= num_per_class: break

# === TRAINING + EVAL for each parameter combo ===
device = torch.device("cpu")
results_summary = []

for i, mel_params in enumerate(param_combos):
    mfcc_params = {"n_mfcc": mel_params["n_mels"], "melkwargs": mel_params}
    tag = f"nfft{mel_params['n_fft']}_hop{mel_params['hop_length']}_win{mel_params['win_length']}_nmels{mel_params['n_mels']}"
    print(f"\n##### RUN {i+1} - Mel params: {mel_params} #####")
    save_sample_images(full_df, num_per_class=2, mel_params=mel_params, mfcc_params=mfcc_params, tag=tag)

    # Split train/test
    train_df, test_df = train_test_split(
        full_df,
        test_size=0.2,
        stratify=full_df["label"],
        random_state=42
    )

    train_dataset = DualInputDataset(train_df, mel_params, mfcc_params, train=True, augment=True)
    test_dataset = DualInputDataset(test_df, mel_params, mfcc_params, train=False, augment=False)
    trainloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    def train_one_epoch(model, loader, optimizer, scheduler, criterion, epoch, device):
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

    def test_eval(model, loader, criterion, device):
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

    # === Fine-tuning ONLY HEAD ===
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
        loss = train_one_epoch(model, trainloader, optimizer, scheduler, criterion, epoch, device)
        train_losses.append(loss)
        acc, _, _ = test_eval(model, testloader, criterion, device)
        test_accuracies.append(acc)

    # === Fine-tuning: HEAD + layer4 ===
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_TAIL)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR_TAIL,
        steps_per_epoch=len(trainloader),
        epochs=EPOCHS_TAIL,
        anneal_strategy='linear'
    )

    print("\n=== FINE-TUNING head + layer4 (last conv block) ===")
    for epoch in range(EPOCHS_TAIL):
        loss = train_one_epoch(model, trainloader, optimizer, scheduler, criterion, EPOCHS_HEAD + epoch, device)
        train_losses.append(loss)
        acc, _, _ = test_eval(model, testloader, criterion, device)
        test_accuracies.append(acc)

    # === Final evaluation: Confusion matrix and classification report ===
    print("\n=== Final evaluation on test set ===")
    _, all_labels, all_preds = test_eval(model, testloader, criterion, device)
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["HC", "PD"])
    plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix {tag}")
    cm_path = f"outputs/results/confusion_matrix_{tag}.png"
    plt.savefig(cm_path)
    plt.close()

    report = classification_report(all_labels, all_preds, target_names=["HC", "PD"])
    report_path = f"outputs/results/classification_report_{tag}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print("Classification report saved:", report_path)

    # === Accuracy & loss graph over epochs ===
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, len(test_accuracies) + 1), test_accuracies, marker='o', label="Test accuracy")
    plt.plot(np.arange(1, len(train_losses) + 1), train_losses, marker='x', label="Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.legend()
    plt.title(f"Accuracy & Loss over epochs\n{tag}")
    plt.grid()
    acc_path = f"outputs/results/learning_curve_{tag}.png"
    plt.savefig(acc_path)
    plt.close()

    # === Summary per combo for quick view ===
    results_summary.append({
        "params": tag,
        "max_acc": max(test_accuracies),
        "final_acc": test_accuracies[-1],
        "final_train_loss": train_losses[-1],
        "matrix_path": cm_path,
        "curve_path": acc_path,
        "report_path": report_path
    })

# === Gridsearch summary ===
print("\n=== Gridsearch summary ===")
for r in results_summary:
    print(f"{r['params']}: max test acc={r['max_acc']*100:.2f}%, final test acc={r['final_acc']*100:.2f}%, ConfMat={r['matrix_path']}, Curve={r['curve_path']}")

print(f"\nTotal time: {(time.time()-start)/60:.2f} minutes")
print("All results have been saved in outputs/ and outputs/results/.")
print("""
Pipeline gridsearch: saves waveform, MelSpectrogram, MFCC as images, confusion matrix, learning curve plots for each parameter combination.
Optimized for modest CPU.
""")
