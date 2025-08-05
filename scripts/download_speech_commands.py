import requests
import tarfile
import os

# Define path-ul unde vrei să fie salvat dataset-ul
output_dir = "./data1/SpeechCommands/"
os.makedirs(output_dir, exist_ok=True)

# Link-ul oficial al dataset-ului
url = "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
output_tar = os.path.join(output_dir, "speech_commands_v0.02.tar.gz")

# Descarcă arhiva
print("Se descarcă Speech Commands v0.02...")
with requests.get(url, stream=True) as r:
    with open(output_tar, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Descărcarea s-a terminat.")

# Dezarhivează arhiva .tar.gz
print("Se dezarhivează (poate dura câteva minute)...")
with tarfile.open(output_tar, "r:gz") as tar:
    tar.extractall(path=output_dir)
print("Dezarhivarea s-a terminat.")

# (Opțional) Șterge arhiva .tar.gz după extragere
os.remove(output_tar)
print("Gata! Dataset-ul e în:", output_dir)