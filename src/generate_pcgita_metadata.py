import os
import pandas as pd

HC_FOLDER = r"C:\Users\claud\PycharmProjects\pytorch_project\pytorch_project\data\PC-GITA_16kHz\PC-GITA_16kHz\monologue\sin normalizar\hc"
PD_FOLDER = r"C:\Users\claud\PycharmProjects\pytorch_project\pytorch_project\data\PC-GITA_16kHz\PC-GITA_16kHz\monologue\sin normalizar\pd"

rows = []
for folder, label, class_name in [
    (HC_FOLDER, 0, "HC"),
    (PD_FOLDER, 1, "PD")
]:
    for file in os.listdir(folder):
        if file.endswith('.wav'):
            full_path = os.path.join(folder, file)
            rows.append({
                "audio_path": full_path,
                "label": label,
                "class": class_name
            })

df = pd.DataFrame(rows)
print("Primele 5 fișiere:", df.head())

# Salvează tot sau doar primele N exemple din fiecare clasă
output_csv = r"data/neurovoz/PC-GITA_16kHz/PCGITA_metadata_monologue.csv"
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
df.to_csv(output_csv, index=False)
print(f"CSV generat: {len(df)} exemple (HC: {(df['label']==0).sum()}, PD: {(df['label']==1).sum()})\nLocație: {output_csv}")