import torch
import torchaudio
import torchvision
import sklearn
import matplotlib
import pandas
import numpy

print(f"PyTorch version: {torch.__version__}")
print(f"torchaudio version: {torchaudio.__version__}")
print(f"torchvision version: {torchvision.__version__}")
print(f"scikit-learn version: {sklearn.__version__}")
print(f"matplotlib version: {matplotlib.__version__}")
print(f"pandas version: {pandas.__version__}")
print(f"numpy version: {numpy.__version__}")

print(f"\nCUDA available (should be False for Intel iGPU): {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("Running PyTorch on CPU. This is expected for Intel integrated graphics.")
else:
    print("CUDA is available. This might be unexpected for Intel iGPU.")

print("\nToate bibliotecile au fost importate cu succes!")