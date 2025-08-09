# PD-Project-

## Project Description

**PD-Project-** is a Python-based project for analyzing and processing speech data, with a focus on supporting Parkinson's Disease research using deep learning. The project leverages neural networks (PyTorch) and data science tools to extract features from speech recordings, handle metadata, and provide reproducible experiments and results.

---

## Project Highlights and Workflow

### 1. Multiple Parameter Experiments

- The pipeline runs for **three different parameter combinations** (e.g., n_fft=400, hop=160, win=400 and similar variants).  
- For each parameter set, the full experiment—preprocessing, training, and evaluation—is repeated.

### 2. Comprehensive Output Saving

- **For every parameter combination, corpus, and label (first two examples per label):**
  - The code saves images for the **waveform**, **MelSpectrogram**, and **MFCC** representations.
- **For each experiment:**
  - The script saves the **confusion matrix** as an image.
  - Generates and saves a **classification report** (precision, recall, f1-score, accuracy).
  - Plots and saves **learning curves** (accuracy and/or loss across epochs).
- **All output files** are organized in `outputs/` and `outputs/results/` for clarity and reproducibility.

### 3. Efficient Resource Utilization

- The workflow is **optimized for modest hardware** (e.g., CPU i5, no GPU):
  - **Batch size is set small** (batch=4).
  - **Number of epochs is low** to ensure quick runs.
  - **Simple data augmentation** is used to avoid heavy computation.
  - All data processing and training steps are tailored for fast execution and low memory consumption.

### 4. Dual Corpus Support and Data Handling

- **Reads and combines data** from both corpora:
  - **Neurovoz**
  - **PC-GITA Monologue**
- **Concatenates all data** into a single DataFrame for joint processing.
- **Performs a stratified train/test split**, ensuring equal class proportions (**HC/PD**) from both corpora in each subset.

### 5. Dual Input Feature Learning

- The model supports **dual input features** (MelSpectrogram + MFCC), enabling richer representations for improved classification.
- Both features are prepared and passed to the model during training and evaluation.

### 6. Training & Evaluation Strategy

- **Batch size remains small** and data augmentation is simple, maintaining compatibility with low-resource hardware.
- **Stratified splitting** guarantees no data leakage and class/corpus balance.
- **All optimizations target efficient CPU-only training.**

### 7. Transfer Learning & Cross-Corpus Experiments

- **Transfer learning/cross-corpus strategies:**  
  - **Train on Neurovoz, test on PC-GITA, and vice versa.**
  - Optionally, fine-tune on both corpora or ensemble predictions from multiple models.
- **Fine-tuning is performed using a staged approach:**
  - First, only the last (classification) layer of the model is trained, allowing rapid adaptation.
  - Then, the entire model is unfrozen and trained for a few more epochs (full fine-tuning).
  - This approach maximizes knowledge transfer while preventing overfitting and reducing compute requirements.

---

## Results and Reproducibility

- For each experiment, you will find:
  - **Sample images** with waveforms, MelSpectrograms, and MFCCs for each class and corpus.
  - **Confusion matrices**, **classification reports**, and **learning curves** in the `outputs/` and `outputs/results/` folders.
  - **Trained model weights** for each experimental setup.
- The design ensures all results are easily reproducible and interpretable, even on modest hardware.

---

## Usage

### 1. Installation

Clone the repository and install the required dependencies:
```bash
git clone https://github.com/ClaudiaAngela/PD-Project-.git
cd PD-Project-
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Data Preparation

- Make sure the audio and metadata files from both **Neurovoz** and **PC-GITA Monologue** corpora are placed in their respective folders as described in the repo structure.
- The combined DataFrame and all preprocessing are handled automatically by the pipeline.

### 3. Running Experiments

Run the main script to launch experiments with all parameter combinations:
```bash
python main.py
```
- By default, this will:
  - Run all parameter combinations (as defined in the script)
  - Save all outputs in `outputs/` and `outputs/results/`

If you want to modify parameter values or experiment settings, edit the relevant sections at the top of `main.py`.

#### Cross-Corpus Transfer Learning Example

To perform transfer learning (e.g., train on Neurovoz, test on PC-GITA), set the experiment mode in `main.py` accordingly. The script is designed to support these scenarios and will output results for each corpus combination.

### 4. Output Files

After completion, check the following folders:
- `outputs/` – contains all images and reports for each experiment and parameter set
- `outputs/results/` – contains additional metrics, logs, and model weights

---
## Contact

For questions or collaboration, please contact [ClaudiaAngela](https://github.com/ClaudiaAngela).
