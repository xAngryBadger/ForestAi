import pandas as pd

def validar_csv(arquivo_csv, label_dict):
    print(f"Carregando CSV: {arquivo_csv}")
    df = pd.read_csv(arquivo_csv)
    
    # Colunas obrigatórias
    colunas_esperadas = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'label']
    for c in colunas_esperadas:
        if c not in df.columns:
            print(f"[ERRO] Coluna '{c}' não encontrada no CSV!")
            return
    
    print("Todas as colunas obrigatórias estão presentes.")
    
    # Verificar valores ausentes
    missing = df.isnull().sum()
    if missing.any():
        print("[ATENÇÃO] Valores ausentes encontrados:")
        print(missing[missing > 0])
    else:
        print("Sem valores ausentes.")
    
    # Verificar validade das caixas delimitadoras
    invalid_boxes = df[(df['xmin'] >= df['xmax']) | (df['ymin'] >= df['ymax']) | 
                       (df['xmin'] < 0) | (df['ymin'] < 0)]
    if not invalid_boxes.empty:
        print(f"[ERRO] Encontradas {len(invalid_boxes)} caixas delimitadoras inválidas (xmin < xmax, ymin < ymax, valores não-negativos):")
        print(invalid_boxes)
    else:
        print("Caixas delimitadoras válidas.")
    
    # Verificar se labels estão no dicionário
    labels_unicos = df['label'].unique()
    print(f"Labels únicos encontrados: {labels_unicos}")
    
    labels_invalidos = [label for label in labels_unicos if label not in label_dict]
    if labels_invalidos:
        print(f"[ERRO] Labels inválidos encontrados: {labels_invalidos}")
    else:
        print("Todos os labels são válidos.")
    
    # Mapear labels para índices e verificar se existem índices fora do intervalo
    df['label_idx'] = df['label'].map(label_dict)
    
    if df['label_idx'].isnull().any():
        print("[ERRO] Existem labels que não foram mapeados corretamente para índices:")
        print(df[df['label_idx'].isnull()])
        return
    
    max_label_idx = max(label_dict.values())
    if df['label_idx'].max() > max_label_idx or df['label_idx'].min() < 0:
        print("[ERRO] Índices das labels fora do intervalo esperado.")
    else:
        print(f"Índices das labels dentro do intervalo esperado (0 a {max_label_idx}).")
    
    # Estatísticas gerais
    print("\nResumo geral das anotações:")
    print(df['label'].value_counts())
    print(f"Total de anotações: {len(df)}")

# Exemplo de uso:
if __name__ == "__main__":
    arquivo_csv = "E:/deepbug ai/model/checkpoints/train_annotations.csv"  # Ajuste para o caminho do seu arquivo
    label_dict = {'fedegoso': 0, 'mudafedegoso': 1}
    
    validar_csv(arquivo_csv, label_dict)
