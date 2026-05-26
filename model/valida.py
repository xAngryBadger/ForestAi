import pandas as pd
import os

# 🔧 CONFIGURAÇÕES
CLASSES_VALIDAS = ["tree", "fedegoso", "mudafedegoso"]  # <- todas em minúsculo
PASTA_IMAGENS = r"E:/deepbug ai/model/fotos"

ARQUIVOS_CSV = [
    r"E:/deepbug ai/model/labels.csv",
    r"E:/deepbug ai/model/checkpoints/train_annotations.csv",
    r"E:/deepbug ai/model/checkpoints/val_annotations.csv"
]

print("=" * 60)
print("🛠️ INICIANDO CORREÇÃO E VALIDAÇÃO DOS CSVs")
print("=" * 60)

for arquivo in ARQUIVOS_CSV:
    print(f"\n📄 Processando: {arquivo}")
    
    if not os.path.exists(arquivo):
        print(f"❌ Arquivo não encontrado: {arquivo}")
        continue

    df = pd.read_csv(arquivo)

    # 🔧 Corrigir nome da coluna de imagem
    if 'image_path' in df.columns:
        df = df.rename(columns={'image_path': 'filename'})
        print("✅ Coluna 'image_path' renomeada para 'filename'.")
    elif 'filename' in df.columns:
        print("✅ Coluna de imagem já é 'filename'.")
    else:
        print("❌ Nenhuma coluna de imagem encontrada (esperado 'filename' ou 'image_path').")
        continue

    # 🔧 Corrigir nome da coluna de classe
    coluna_label = None
    for c in df.columns:
        if c.lower() in ["class", "label", "tree"]:  # covers vários padrões
            coluna_label = c
            break
    if coluna_label is None:
        print("❌ Coluna de classe não encontrada!")
        continue
    else:
        print(f"✅ Coluna de classe detectada: '{coluna_label}'")

    # 🔧 Converter labels para minúsculo (padronização)
    df[coluna_label] = df[coluna_label].str.strip().str.lower()

    # 🔍 Validar labels
    labels_encontradas = set(df[coluna_label].unique())
    labels_invalidas = labels_encontradas - set(CLASSES_VALIDAS)

    if labels_invalidas:
        print(f"❌ Labels inválidas encontradas: {labels_invalidas}")
    else:
        print("✅ Todas as labels estão corretas.")

    # 🔍 Validar bounding boxes
    bbox_invalidas = df[
        (df['xmin'] >= df['xmax']) |
        (df['ymin'] >= df['ymax'])
    ]
    if not bbox_invalidas.empty:
        print(f"❌ {len(bbox_invalidas)} bounding boxes inválidas encontradas!")
    else:
        print("✅ Bounding boxes válidas.")

    # 🔍 Validar existência das imagens
    imagens_ausentes = []
    for img in df['filename'].unique():
        caminho = os.path.join(PASTA_IMAGENS, img)
        if not os.path.exists(caminho):
            imagens_ausentes.append(img)
    if imagens_ausentes:
        print(f"❌ {len(imagens_ausentes)} imagens não encontradas na pasta.")
        for img in imagens_ausentes:
            print(f"   - {img}")
    else:
        print("✅ Todas as imagens estão presentes na pasta.")

    # 💾 Salvar CSV corrigido
    df.to_csv(arquivo, index=False)
    print(f"💾 CSV corrigido e salvo: {arquivo}")

print("\n" + "=" * 60)
print("✅ PROCESSO FINALIZADO")
print("=" * 60)
