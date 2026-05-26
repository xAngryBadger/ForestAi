import os
print(f"Diretório de trabalho atual: {os.getcwd()}")

import pandas as pd
from deepforest import main
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import glob
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

# 1. Define paths to your data
image_dir = r"C:\Hololeuca\hololeuca img\semiaerea40"
annotations_csv = r"C:\Hololeuca\hololeuca csv beta\labels.csv"
saved_model = r"C:\Hololeuca\hololeuca modelo beta"
train_csv_path = os.path.join(saved_model, "train_annotations.csv") # Caminho para o CSV de treino
test_csv_path = os.path.join(saved_model, "test_annotations.csv") # Caminho para o CSV de teste

# 2. Load the annotations
with open(annotations_csv, 'r', encoding='utf-8') as f:
    lines = f.readlines()
print(f"Número de linhas lidas diretamente do arquivo CSV: {len(lines)}")

df = pd.read_csv(annotations_csv, delimiter=',', encoding='utf-8', header=0)
print(f"Número de linhas no DataFrame df após a leitura do CSV: {len(df)}")

# Create absolute paths for image_path
def create_absolute_path(row, image_dir):
    return os.path.join(image_dir, os.path.basename(row["filename"]))

df["image_path"] = df.apply(lambda row: create_absolute_path(row, image_dir), axis=1)

# Renomear a coluna 'Tree' para 'label'
if 'Tree' in df.columns:
    df = df.rename(columns={'Tree': 'label'})
else:
    print("A coluna 'Tree' não foi encontrada para renomear para 'label'.")

# 3. Split the data into training and testing sets
train_df, test_df = train_test_split(df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']], test_size=0.1, random_state=42)

# Salvar o train_df em um arquivo CSV temporário
train_df.to_csv(train_csv_path, index=False)
print(f"Arquivo CSV de treino salvo em: {train_csv_path}")

# Imprimindo o tamanho do train_df
print(f"Número de imagens no train_df para treinamento: {len(train_df)}")
print(f"Batch size configurado para treinamento: {4}") # Assumindo batch_size = 4
print(f"Número esperado de batches por época (aproximadamente): {len(train_df) / 4}")
print(f"Número de imagens no test_df para avaliação: {len(test_df)}")

# Salvar o test_df em um arquivo CSV temporário para avaliação
test_df.to_csv(test_csv_path, index=False)
print(f"Arquivo CSV de teste salvo em: {test_csv_path}")

# 4. Initialize the DeepForest model
model = main.deepforest(config_args={
    "train": {
        "scheduler": {
            "type": "ReduceLROnPlateau",
            "params": {
                "mode": "min",
                "factor": 0.1,
                "patience": 10,
                "threshold": 0.0001,
                "threshold_mode": "rel",
                "cooldown": 0,
                "min_lr": 0,
                "eps": 1e-08
            }
        },
        "lr": 0.0005,
        "csv_file": train_csv_path,
        "root_dir": image_dir,
        "epochs": 40,
        "batch_size": 4,
        "fast_dev_run": False,
        "preload_images": True
    }
})

# Inicializar um logger do TensorBoard
logger = TensorBoardLogger("lightning_logs")

# Inicializar o trainer do PyTorch Lightning explicitamente
trainer = pl.Trainer(max_epochs=model.config['train']['epochs'], logger=logger, accelerator='auto', devices=1 if not os.environ.get("CUDA_VISIBLE_DEVICES") else 1)

# Train the model using the trainer
print("Iniciando o treinamento do modelo...")
trainer.fit(model, train_dataloaders=model.train_dataloader())
print("Treinamento do modelo concluído.")

print("Dicionário de rótulos do modelo após o treinamento:")
print(model.label_dict)

# Avaliação
print(f"Valor de image_dir: {image_dir}")
print(f"O diretório existe? {os.path.isdir(image_dir)}")
# Avaliação usando o caminho para o CSV de teste
print(f"Número de imagens no test_df para avaliação: {len(test_df)}")
results = model.evaluate(test_csv_path, root_dir=image_dir)
print(results)

# Verificar se o modelo faz alguma predição nas imagens de teste
test_images = test_df['image_path'].tolist()[:5]
all_predictions = []
for img_path in test_images:
    image = plt.imread(img_path)
    preds = model.predict_image(image=image)
    print(f"Predições para {os.path.basename(img_path)}:\n{preds}")
    all_predictions.append(preds)

# 7. Make predictions (predição da imagem única)
image_path_predict = r'C:\Hololeuca\hololeuca img\testing\teste1img\semi4.jpg'
image_to_predict = plt.imread(image_path_predict)
predictions = model.predict_image(image=image_to_predict)

# 8. Visualize
image = plt.imread(image_path_predict)
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
ax.imshow(image)

if predictions is not None and not predictions.empty:
    for index, row in predictions.iterrows():
        xmin = int(row['xmin'])
        ymin = int(row['ymin'])
        xmax = int(row['xmax'])
        ymax = int(row['ymax'])
        label = row['label']
        color = 'r'
        ax.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        )
        ax.text(xmin, ymin - 5, label, color=color, fontsize=12)

plt.show()

# Limpar os arquivos CSV temporários
if os.path.exists(train_csv_path):
    os.remove(train_csv_path)
if os.path.exists(test_csv_path):
    os.remove(test_csv_path)