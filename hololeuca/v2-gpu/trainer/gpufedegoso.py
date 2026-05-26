import os
import pandas as pd
from deepforest import main
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch # Importar torch para verificar GPU

# 1. Define paths to your data
# Caminhos atualizados para Fedegoso (mantidos como você solicitou)
image_dir = r"C:/HololeucaGPU/hololeuca img/fedegoso"
annotations_csv = r"C:/HololeucaGPU/hololeuca csv beta/labels.csv" # Mantenha o CSV original
saved_model = r"C:/HololeucaGPU/hololeuca modelo beta/lightning_logs" # Diretório de saída para o modelo treinado e CSVs temporários
train_csv_path = os.path.join(saved_model, "train_annotations_fedegoso.csv")
val_csv_path = os.path.join(saved_model, "val_annotations_fedegoso.csv")

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

# Renomear a coluna 'Tree' para 'label' (REVERTIDO)
# Agora busca por 'Tree' como no código original para Cecropia Hololeuca
if 'Tree' in df.columns:
    df = df.rename(columns={'Tree': 'label'})
else:
    # Este print será ativado se 'Tree' não for encontrado e 'Fedegoso' também não.
    # Se seu CSV tem 'Fedegoso' e você quer que o modelo use isso, você precisará
    # ajustar este if/elif ou garantir que a coluna 'Tree' exista.
    print("A coluna 'Tree' não foi encontrada para renomear para 'label'. Verifique seu CSV.")


print("Rótulos únicos encontrados no DataFrame 'df' após leitura e renomeação:")
print(df['label'].unique())

# Filtrar o DataFrame para incluir apenas as anotações de 'Cecropia Hololeuca' (REVERTIDO)
# Este filtro *não* é para Fedegoso. Se você está treinando Fedegoso,
# o valor da coluna 'label' deve ser 'Senna obtusifolia'.
# Se o CSV ainda contém 'Cecropia Hololeuca', este filtro é apropriado.
# Se o CSV contém 'Senna obtusifolia', este filtro retornará um DataFrame vazio.
df_filtered = df[df['label'] == 'Fedegoso'].copy() # Revertido para Cecropia Hololeuca

if df_filtered.empty:
    print("AVISO: Nenhuma anotação para 'Fedegoso' encontrada no CSV após a filtragem. Verifique se o rótulo está correto no CSV.")
else:
    print(f"Número de anotações de 'Cecropia Hololeuca' encontradas: {len(df_filtered)}")
    print("Exemplo das primeiras 5 linhas do DataFrame filtrado para 'Fedegoso':")
    print(df_filtered.head())


# 3. Split the data into training and validation sets (usando df_filtered)
# Se df_filtered estiver vazio, esta parte irá falhar ou criar dataframes vazios.
if not df_filtered.empty:
    train_df, val_df = train_test_split(df_filtered[['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']], test_size=0.1, random_state=42)
else:
    print("Pulando a divisão de dados, pois 'df_filtered' está vazio.")
    train_df = pd.DataFrame()
    val_df = pd.DataFrame()

# Salvar o train_df em um arquivo CSV temporário
if not train_df.empty:
    train_df.to_csv(train_csv_path, index=False)
    print(f"Arquivo CSV de treino para Fedegoso salvo em: {train_csv_path}")
else:
    print("Nenhum arquivo CSV de treino salvo, pois train_df está vazio.")


# Definir batch_size
BATCH_SIZE = 8
# Com a GPU, você pode tentar aumentar o BATCH_SIZE para 16 ou até 32,
# dependendo da sua VRAM (4GB da GTX 1050 Ti podem suportar até 16 ou 32 para DeepForest).
# Se der erro de memória, reduza.
print(f"Número de imagens no train_df para treinamento: {len(train_df)}")
print(f"Batch size configurado para treinamento: {BATCH_SIZE}")
if BATCH_SIZE > 0:
    print(f"Número esperado de batches por época (aproximadamente): {len(train_df) / BATCH_SIZE}")
print(f"Número de imagens no val_df para validação: {len(val_df)}")

# Salvar o val_df em um arquivo CSV temporário para validação
if not val_df.empty:
    val_df.to_csv(val_csv_path, index=False)
    print(f"Arquivo CSV de validação para Fedegoso salvo em: {val_csv_path}")
else:
    print("Nenhum arquivo CSV de validação salvo, pois val_df está vazio.")

# 4. Define label dictionary (REVERTIDO)
label_dict = {"Fedegoso": 0} # Revertido para Cecropia Hololeuca

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
        "epochs": 120,
        "batch_size": BATCH_SIZE,
        "fast_dev_run": False,
        "preload_images": True,
        "num_classes": 1, # Apenas uma classe: Cecropia Hololeuca
        "label_dict": label_dict,
        "num_workers": 4 # Mantido para otimizar carregamento de dados
    },
    "validation": {
        "csv_file": val_csv_path,
        "root_dir": image_dir,
        "batch_size": BATCH_SIZE,
        "num_workers": 2 # Mantido para otimizar carregamento de dados
    }
})

model.label_dict = label_dict
print("Dicionário de rótulos do modelo (forçado após inicialização):")
print(model.label_dict)

# 6. Initialize a TensorBoard logger
logger = TensorBoardLogger(saved_model) # Usar o caminho ajustado para o logger

# 7. Initialize the PyTorch Lightning trainer explicitly
trainer = pl.Trainer(
    max_epochs=model.config['train']['epochs'],
    logger=logger,
    accelerator='gpu', # MUDANÇA PRINCIPAL: De 'cpu' para 'gpu'
    devices=1,           # Para GPU, devices=1 para usar uma GPU (a 1050 Ti)
    # Se você quiser monitorar o progresso visualmente (e talvez o tempo):
    # enable_progress_bar=True,
    # log_every_n_steps=50 # Logar a cada 50 passos
)

# NOVO: Salvar o modelo treinado (para não perder o progresso)
# Garanta que a pasta 'saved_model' exista
os.makedirs(saved_model, exist_ok=True)
model_save_path = os.path.join(saved_model, "deepforest_fedegoso_trained_gpu.pt") # Mantido o nome do modelo para Fedegoso

# 8. Train the model
print("Iniciando o treinamento do modelo na GPU...")
if not train_df.empty and not val_df.empty:
    try:
        trainer.fit(model)
        print("Treinamento do modelo concluído.")
        # Salvar o modelo treinado após o fit
        model.save_model(model_save_path)
        print(f"Modelo treinado salvo em: {model_save_path}")
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        print("Verifique se há dados suficientes nos dataframes de treino e validação, e se os caminhos estão corretos.")
else:
    print("Não foi possível iniciar o treinamento: DataFrames de treino ou validação estão vazios.")


print("Dicionário de rótulos do modelo após o treinamento:")
print(model.label_dict)

# 9. Avaliação
print(f"Valor de image_dir: {image_dir}")
print(f"O diretório existe? {os.path.isdir(image_dir)}")
print(f"Número de imagens no val_df (usado para teste/avaliação final): {len(val_df)}")
if not val_df.empty and os.path.exists(val_csv_path):
    results = model.evaluate(val_csv_path, root_dir=image_dir)
    print("Resultados da avaliação:")
    print(results)
else:
    print("Não foi possível realizar a avaliação: DataFrame de validação ou CSV de validação não existem ou estão vazios.")

# 10. Verify if the model makes predictions on test images
if not val_df.empty:
    test_images_for_predict = val_df['image_path'].tolist()[:5]
    all_predictions = []
    print("\nRealizando predições em algumas imagens de teste:")
    for img_path in test_images_for_predict:
        try:
            image = plt.imread(img_path)
            # Certifique-se de que o modelo está no modo de avaliação e na GPU
            model.eval()
            if torch.cuda.is_available():
                model.to("cuda") # Move o modelo para a GPU para predição
            preds = model.predict_image(image=image)
            print(f"Predições para {os.path.basename(img_path)}:\n{preds}")
            all_predictions.append(preds)
        except FileNotFoundError:
            print(f"AVISO: Imagem não encontrada para predição: {img_path}. Pulando.")
        except Exception as e:
            print(f"Erro ao predizer em {img_path}: {e}")
else:
    print("Não há imagens de validação para realizar predições de teste.")

# 11. Make predictions on a single image and visualize
# ATENÇÃO: Verifique se este caminho de imagem também está na nova estrutura de pastas
# Escolha uma imagem de teste que você saiba que contém Fedegoso
image_path_predict = r'C:\HololeucaGPU\hololeuca img\fedegoso\DJI_0856.jpg' # << SUBSTITUA POR UM CAMINHO VÁLIDO
if os.path.exists(image_path_predict):
    print(f"\nRealizando predição e visualização em uma única imagem: {os.path.basename(image_path_predict)}")
    image_to_predict = plt.imread(image_path_predict)
    # Certifique-se de que o modelo está no modo de avaliação e na GPU
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda") # Move o modelo para a GPU para predição
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

    # O título da imagem de predição foi ajustado para refletir o rótulo usado no label_dict
    plt.title(f"Fedegoso em: {os.path.basename(image_path_predict)}")
    plt.show()
else:
    print(f"AVISO: A imagem de teste para visualização '{image_path_predict}' não foi encontrada. Por favor, atualize o caminho.")


# 12. Clean up temporary CSV files
print("\nLimpando arquivos CSV temporários...")
if os.path.exists(train_csv_path):
    os.remove(train_csv_path)
    print(f"Removido: {train_csv_path}")
if os.path.exists(val_csv_path):
    os.remove(val_csv_path)
    print(f"Removido: {val_csv_path}")
print("Limpeza concluída.")