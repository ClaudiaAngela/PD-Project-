import os
import pandas as pd

def prepare_neurovoz_dataframe(hc_csv, pd_csv):
    hc_df = pd.read_csv(hc_csv)
    pd_df = pd.read_csv(pd_csv)
    hc_df["label"] = 0
    pd_df["label"] = 1
    hc_df = hc_df[["Audio", "label"]].rename(columns={"Audio": "audio_path"})
    pd_df = pd_df[["Audio", "label"]].rename(columns={"Audio": "audio_path"})
    df = pd.concat([hc_df, pd_df], ignore_index=True)
    # Transforma calea in absolută
    project_root = os.path.dirname(os.path.abspath(__file__))
    df["audio_path"] = df["audio_path"].apply(lambda x: os.path.normpath(os.path.join(project_root, x)))
    # Păstrează doar fișierele existente
    df = df[df["audio_path"].apply(os.path.isfile)].reset_index(drop=True)
    return df