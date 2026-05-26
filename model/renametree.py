import pandas as pd
import os

# Caminho dos CSVs
csv_files = [
    "E:/deepbug ai/model/labels.csv",
    "E:/deepbug ai/model/checkpoints/train_annotations.csv",
    "E:/deepbug ai/model/checkpoints/val_annotations.csv"
]

for file in csv_files:
    if os.path.exists(file):
        df = pd.read_csv(file)

        # Verificar se 'Tree' está presente
        if 'Tree' in df.columns:
            df = df.rename(columns={'Tree': 'label'})
            df.to_csv(file, index=False)
            print(f"[✔️] Corrigido: {file}")
        else:
            print(f"[⚠️] Já estava correto: {file}")
    else:
        print(f"[❌] Arquivo não encontrado: {file}")
