# PD-Project-

## Project Description
Image Deep Learning Networks Fine-Tuning for Parkinson’s

Disease Speech Applications
**PD-Project-** is a Python-based application aimed at analyzing and processing speech data, with a focus on supporting research into Parkinson's Disease.
The project leverages neural networks (PyTorch) and various data science tools to extract features from speech recordings, handle metadata, and provide reproducible 
experiments and results for academic or clinical purposes.

## Technologies Used

- Python 3.x
- PyTorch
- NumPy, pandas
- (add others if relevant: scikit-learn, matplotlib, etc.)

## Project Structure

- `src/` – core modules and utilities:
  - `dataloader.py` – Loads and preprocesses data.
  - `generate_pcgita_metadata.py` – Generates metadata for the PC-GITA dataset.
  - `interface.py` – Interface functions for processing and prediction.
  - `model.py` – Neural network model definitions.
  - `prepare_dataframe.py` – Prepares dataframes from raw input.
  - `train_utils.py` – Training utilities.
  - `transforms.py` – Data transformation routines.
  - `__init__.py` – Marks the folder as a Python package.
  - `data/neurovoz/PC-GITA_16kHz/PCGITA_metadata_monologue.csv` – Example metadata file.

- `test_script.py` – Script for testing the model and pipeline.
- `.gitignore` – Lists files and folders to be excluded from versioning.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/ClaudiaAngela/PD-Project-.git
    cd PD-Project-
    ```
2. (Optional) Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Modify the command below according to your data and configuration:

```bash
python main.py --input src/data/neurovoz/PC-GITA_16kHz/PCGITA_metadata_monologue.csv --output results.csv
```

Or test the scripts individually:

```bash
python test_script.py
```

## Dataset

The project uses the both  NeuroVoz and PC_GITA_16kHz dataset for speech analysis related to Parkinson's Disease.  
**Note:** The dataset files are not included. Please download and place them in the specified folders as described in the documentation.

## Contribution

Contributions are welcome! Please open issues or pull requests for suggestions, bug reports, or improvements.


## Contact

For questions or collaboration, please contact [ClaudiaAngela](https://github.com/ClaudiaAngela).
