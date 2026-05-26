import os
import pandas as pd
from deepforest import main
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch

# ==== 1. Definição dos diretórios ====
base_dir = r"E:\deepbug ai\model"
image_dir = os.path.join(base_dir, "fotos")
annotations_csv = os.path.join(base_dir, "labels.csv")
checkpoint_dir = os.path.join(base_dir, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

train_csv_path = os.path.join(checkpoint_dir, "train_annotations.csv")
val_csv_path   = os.path.join(checkpoint_dir, "val_annotations.csv")
saved_model_path = os.path.join(checkpoint_dir, "deepforest_multispecies.pt")

# ==== 2. Carregar e preparar anotações ====
df = pd.read_csv(annotations_csv, encoding="utf-8")
df.columns = df.columns.str.strip()
df = df.rename(columns={'Tree': 'label'})  # renomeia Tree→label
df["image_path"] = df["filename"].apply(lambda x: os.path.join(image_dir, os.path.basename(x)))

print(">>> Colunas:", df.columns.tolist())
print(">>> Labels originais:", df["label"].unique())

# ==== 3. Criar e validar label_dict ====
labels = sorted(df["label"].unique())
label_dict = {lbl: idx for idx, lbl in enumerate(labels)}
num_classes = len(label_dict)
print(">>> label_dict:", label_dict)

# Verifica se todo label está no label_dict
invalid = set(df["label"]) - set(label_dict.keys())
if invalid:
    raise ValueError(f"Labels indefinidos no dicionário: {invalid}")

# ==== 4. Split treino/val ====
train_df, val_df = train_test_split(
    df[["image_path","xmin","ymin","xmax","ymax","label"]],
    test_size=0.2, random_state=42, stratify=df["label"]
)
train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)
print(f">>> Train/Val splits gerados: {len(train_df)}/{len(val_df)}")

# ==== 5. Validação extra de índices antes do treino ====
# Vamos mapear manualmente a coluna 'label' para índices e checar
train_df["label_idx"] = train_df["label"].map(label_dict)
if train_df["label_idx"].isnull().any():
    bad = train_df[train_df["label_idx"].isnull()]
    raise ValueError(f"Labels sem mapeamento em train_df:\n{bad[['label']].drop_duplicates()}")
mx = train_df["label_idx"].max()
mn = train_df["label_idx"].min()
assert 0 <= mn <= mx < num_classes, f"Indices out of bounds: min={mn}, max={mx}, num_classes={num_classes}"
print(f">>> Train labels index OK (0 ≤ idx ≤ {num_classes-1})")

# ==== 6. Configuração do modelo DeepForest ====
BATCH_SIZE = 4
cfg = {
    "train": {
        "csv_file": train_csv_path,
        "root_dir": image_dir,
        "epochs": 120,
        "batch_size": BATCH_SIZE,
        "lr": 0.0005,
        "preload_images": False,      # desativo para não mascarar erros
        "fast_dev_run": False,
        "num_classes": num_classes,
        "label_dict": label_dict,
        "num_workers": 2
    },
    "validation": {
        "csv_file": val_csv_path,
        "root_dir": image_dir,
        "batch_size": BATCH_SIZE,
        "num_workers": 2
    }
}
model = main.deepforest(config_args=cfg)
model.label_dict = label_dict

# ==== 7. Logger & Trainer no CPU (debug) ====
logger = TensorBoardLogger(checkpoint_dir, name="debug_logs")
trainer = pl.Trainer(
    max_epochs=cfg["train"]["epochs"],
    logger=logger,
    accelerator="cpu",   # roda no CPU para debugar
    devices=1,
    enable_progress_bar=True,
    fast_dev_run=False
)

# ==== 8. Treinamento ====
print(">>> Iniciando treinamento (CPU, modo debug)...")
trainer.fit(model)
print(">>> Treinamento concluído.")

# ==== 9. (Opcional) Mova para GPU depois de debug: ====
# Se tudo passou no CPU, volte `accelerator='gpu'` e `preload_images=True`.

# ==== 10. Salvar modelo ====
model.save_model(saved_model_path)
print(f">>> Modelo salvo em: {saved_model_path}")
