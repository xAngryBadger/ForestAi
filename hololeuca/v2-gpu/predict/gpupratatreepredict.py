import os
from deepforest import main
import matplotlib.pyplot as plt
import pandas as pd
import pytorch_lightning as pl # Importante para carregar o checkpoint

# 1. Definir o caminho para o seu checkpoint salvo
# **ATENÇÃO: Este caminho foi ajustado com base na sua última informação**
checkpoint_path = r"C:\Hololeuca\hololeuca modelo beta\lightning_logs\lightning_logs\version_20\checkpoints\epoch=119-step=4320.ckpt"

# 2. Caminho para as novas imagens que você quer prever
new_image_dir = r"C:\Hololeuca\hololeuca img\testing\testpostreino0" # Caminho fornecido por você
# Lista para armazenar os caminhos das imagens na pasta
new_image_paths = []
if os.path.exists(new_image_dir):
    for filename in os.listdir(new_image_dir):
        # Filtra apenas arquivos de imagem comuns
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            new_image_paths.append(os.path.join(new_image_dir, filename))
else:
    print(f"Erro: Diretório '{new_image_dir}' não encontrado. Por favor, verifique o caminho.")
    exit() # Sai do script se o diretório não existir

if not new_image_paths:
    print(f"Nenhuma imagem encontrada em '{new_image_dir}'. Por favor, adicione imagens para predição.")
    exit() # Sai do script se não houver imagens

# 3. Definir o dicionário de rótulos (precisa ser o mesmo usado no treinamento)
label_dict = {"Cecropia Hololeuca": 0}

# 4. Inicializar o modelo DeepForest e carregar o checkpoint
loaded_model = main.deepforest.load_from_checkpoint(
    checkpoint_path,
    config_args={"train": {"num_classes": 1, "label_dict": label_dict}}
)
loaded_model.label_dict = label_dict

print(f"Modelo carregado de checkpoint: {checkpoint_path}")
print(f"Dicionário de rótulos do modelo carregado: {loaded_model.label_dict}")
print(f"Pronto para fazer predições em {len(new_image_paths)} imagens.")

# 5. Fazer predições e visualizar para cada nova imagem
for img_path in new_image_paths:
    print(f"\nProcessando imagem: {os.path.basename(img_path)}")
    try:
        image = plt.imread(img_path)
    except Exception as e:
        print(f"Erro ao carregar imagem {os.path.basename(img_path)}: {e}. Pulando.")
        continue

    # Fazer a predição
    predictions = loaded_model.predict_image(image=image)

    if predictions is not None and not predictions.empty:
        print(f"Predições encontradas para {os.path.basename(img_path)}:")
        print(predictions) # Imprime o DataFrame de predições

        # Visualizar as predições
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        for index, row in predictions.iterrows():
            xmin = int(row['xmin'])
            ymin = int(row['ymin'])
            xmax = int(row['xmax'])
            ymax = int(row['ymax'])
            label = row['label']
            score = row['score']
            color = 'g' # Cor da caixa (verde)

            ax.add_patch(
                plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=color, linewidth=2)
            )
            ax.text(xmin, ymin - 5, f'{label}: {score:.2f}', color='white', fontsize=10, bbox=dict(facecolor=color, alpha=0.7))

        plt.title(f"Predições para {os.path.basename(img_path)}")
        plt.axis('off') # Remove os eixos para uma visualização mais limpa
        plt.show() # Exibe a imagem com predições
    else:
        print(f"Nenhuma predição encontrada para {os.path.basename(img_path)}.")

print("\nProcessamento de novas imagens concluído.")