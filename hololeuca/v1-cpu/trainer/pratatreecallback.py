import os
import pandas as pd
from deepforest import main
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
# Removendo a importação do EarlyStopping, pois não o usaremos.
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks.early_stopping import EarlyStopping # Removido

# 1. Define paths to your data
image_dir = r"C:\Hololeuca\hololeuca img\semiaerea40"
annotations_csv = r"C:\Hololeuca\hololeuca csv beta\labels.csv"
saved_model = r"C:\Hololeuca\hololeuca modelo beta"
train_csv_path = os.path.join(saved_model, "train_annotations.csv")
val_csv_path = os.path.join(saved_model, "val_annotations.csv")

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

print("Rótulos únicos encontrados no DataFrame 'df' após leitura e renomeação:")
print(df['label'].unique())

# 3. Split the data into training and validation sets
train_df, val_df = train_test_split(df[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']], test_size=0.1, random_state=42)

# Salvar o train_df em um arquivo CSV temporário
train_df.to_csv(train_csv_path, index=False)
print(f"Arquivo CSV de treino salvo em: {train_csv_path}")

# Definir batch_size aqui para que a mensagem de print o reflita
BATCH_SIZE = 8 # Aumentado o batch size para tentar usar mais RAM e CPU
print(f"Número de imagens no train_df para treinamento: {len(train_df)}")
print(f"Batch size configurado para treinamento: {BATCH_SIZE}")
print(f"Número esperado de batches por época (aproximadamente): {len(train_df) / BATCH_SIZE}")
print(f"Número de imagens no val_df para validação: {len(val_df)}")

# Salvar o val_df em um arquivo CSV temporário para validação
val_df.to_csv(val_csv_path, index=False)
print(f"Arquivo CSV de validação salvo em: {val_csv_path}")

# 4. Define label dictionary
label_dict = {"Cecropia Hololeuca": 0}

# 5. Initialize the DeepForest model
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
        "epochs": 120, # Manter 120 épocas
        "batch_size": BATCH_SIZE, # Usar o novo BATCH_SIZE
        "fast_dev_run": False,
        "preload_images": True,
        "num_classes": 1,
        "label_dict": label_dict,
        "num_workers": 4 # Adicionado para usar mais núcleos da CPU no carregamento de dados
    },
    "validation": {
        "csv_file": val_csv_path,
        "root_dir": image_dir,
        "batch_size": BATCH_SIZE, # Usar o novo BATCH_SIZE para validação também
        "num_workers": 2 # Adicionado para usar mais núcleos da CPU na validação
    }
})

model.label_dict = label_dict
print("Dicionário de rótulos do modelo (forçado após inicialização):")
print(model.label_dict)

# 6. Initialize a TensorBoard logger
logger = TensorBoardLogger("lightning_logs")

# 7. Initialize the PyTorch Lightning trainer explicitly
trainer = pl.Trainer(
    max_epochs=model.config['train']['epochs'],
    logger=logger,
    accelerator='cpu', # Forçar o uso da CPU explicitamente
    devices=1, # Para CPU, devices=1 é o padrão e suficiente
    # callbacks=[] # Removido o argumento callbacks, pois não temos nenhum
)

# 8. Train the model
print("Iniciando o treinamento do modelo para Cecropia Hololeuca...")
trainer.fit(model)
print("Treinamento do modelo concluído.")

print("Dicionário de rótulos do modelo após o treinamento:")
print(model.label_dict)

# 9. Avaliação
print(f"Valor de image_dir: {image_dir}")
print(f"O diretório existe? {os.path.isdir(image_dir)}")
print(f"Número de imagens no val_df (usado para teste/avaliação final): {len(val_df)}")
results = model.evaluate(val_csv_path, root_dir=image_dir)
print(results)

# 10. Verify if the model makes predictions on test images
test_images_for_predict = val_df['image_path'].tolist()[:5]
all_predictions = []
for img_path in test_images_for_predict:
    image = plt.imread(img_path)
    preds = model.predict_image(image=image)
    print(f"Predições para {os.path.basename(img_path)}:\n{preds}")
    all_predictions.append(preds)

# 11. Make predictions on a single image and visualize
image_path_predict = r'C:\Hololeuca\hololeuca img\testing\teste1img\semi4.jpg'
image_to_predict = plt.imread(image_path_predict)
predictions = model.predict_image(image=image_to_predict)

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
        score = row['score']
        color = 'g'
        ax.add_patch(
            plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        )
        ax.text(xmin, ymin - 5, f'{label}: {score:.2f}', color=color, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

plt.title(f"Predição de Cecropia Hololeuca em: {os.path.basename(image_path_predict)}")
plt.show()

# 12. Clean up temporary CSV files
if os.path.exists(train_csv_path):
    os.remove(train_csv_path)
if os.path.exists(val_csv_path):
    os.remove(val_csv_path)