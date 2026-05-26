import os
print(f"Diretório de trabalho atual: {os.getcwd()}")

import pandas as pd
from deepforest import main
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.patches as patches
from deepforest.dataset import TreeDataset
from torch.utils.data import DataLoader
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

def custom_collate_fn_with_category_ids(batch):
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    padded_images = []
    new_targets = []
    max_height = 0
    max_width = 0
    for image in images:
        max_height = max(max_height, image.shape[1])
        max_width = max(max_width, image.shape[2])

    for i, image in enumerate(images):
        pad_h = max_height - image.shape[1]
        pad_w = max_width - image.shape[2]
        padded_image = torch.nn.functional.pad(image, (0, pad_w, 0, pad_h), 'constant', 0)
        padded_images.append(padded_image)

        target = targets[i]
        labels = target['labels']
        category_ids = [label_dict.get(label, -1) for label in labels]
        new_targets.append({'bboxes': target['bboxes'], 'labels': labels, 'category_ids': category_ids})

    padded_images = torch.stack(padded_images)
    return padded_images, new_targets

# 1. Define paths to your data
image_dir = r"C:\Hololeuca\hololeuca img\semiaerea40"
annotations_csv = r"C:\Hololeuca\hololeuca csv beta\labels.csv"
saved_model = r"C:\Hololeuca\hololeuca modelo beta"
train_csv_path = os.path.join(saved_model, "train_annotations.csv")
test_csv_path = os.path.join(saved_model, "test_annotations.csv")

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

# Salvar os DataFrames de treino e teste em arquivos CSV
train_df.to_csv(train_csv_path, index=False)
print(f"Arquivo CSV de treino salvo em: {train_csv_path}")

test_df.to_csv(test_csv_path, index=False)
print(f"Arquivo CSV de teste salvo em: {test_csv_path}")

# 4. Define label dictionary
label_dict = {"Cecropia Hololeuca": 0}

# 5. Define transformations (tentativa com 'label' como label_field)
train_transforms = A.Compose([
    A.Resize(height=512, width=640),
    ToTensorV2()
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['label']))

# 6. Create TreeDataset for training with transformations
train_dataset = TreeDataset(
    csv_file=train_csv_path,
    root_dir=image_dir,
    label_dict=label_dict,
    train=True,
    preload_images=True,
    transforms=train_transforms
)

# 7. Create DataLoader for training (usando a função de collate personalizada)
train_dataloader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0,
    collate_fn=custom_collate_fn_with_category_ids
)

# 8. Initialize the DeepForest model
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
        "preload_images": True,
        "num_classes": 1,
        "label_dict": label_dict
    }
})

# 9. Initialize a dummy label_dict in the model if it's None
if model.label_dict is None:
    model.label_dict = label_dict

# Inicializar um logger do TensorBoard
logger = TensorBoardLogger("lightning_logs")

# Inicializar o trainer do PyTorch Lightning explicitamente
trainer = pl.Trainer(max_epochs=model.config['train']['epochs'], logger=logger, accelerator='auto', devices=1 if not os.environ.get("CUDA_VISIBLE_DEVICES") else 1)

# Train the model using the custom DataLoader
print("Iniciando o treinamento do modelo para Cecropia Hololeuca...")
trainer.fit(model, train_dataloaders=train_dataloader)
print("Treinamento do modelo concluído.")

print("Dicionário de rótulos do modelo após o treinamento:")
print(model.label_dict)

# Avaliação (precisaremos aplicar transformações semelhantes ao dataset de teste se quisermos avaliar corretamente)
print(f"Valor de image_dir: {image_dir}")
print(f"O diretório existe? {os.path.isdir(image_dir)}")
print(f"Número de imagens no test_df para avaliação: {len(test_df)}")
results = model.evaluate(test_csv_path, root_dir=image_dir)
print(results)

# Visualização (a predição em uma única imagem não precisa de redimensionamento prévio)
test_images = test_df['image_path'].tolist()[:5]
all_predictions = []
for img_path in test_images:
    image = plt.imread(img_path)
    preds = model.predict_image(image=image, root_dir=image_dir)
    print(f"Predições para {os.path.basename(img_path)}:\n{preds}")
    all_predictions.append(preds)

image_path_predict = r'C:\Hololeuca\hololeuca img\testing\teste1img\semi4.jpg'
image_to_predict = plt.imread(image_path_predict)
predictions = model.predict_image(image=image_to_predict, root_dir=image_dir)

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
        color = 'g'
        ax.add_patch(
            patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
        )
        ax.text(xmin, ymin - 5, f'{label}: {row["score"]:.2f}', color=color, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

plt.title(f"Predição de Cecropia Hololeuca em: {os.path.basename(image_path_predict)}")
plt.show()

# Limpar arquivos temporários
if os.path.exists(train_csv_path):
    os.remove(train_csv_path)
if os.path.exists(test_csv_path):
    os.remove(test_csv_path)