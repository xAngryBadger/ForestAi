# ForestAI — Detecção de Espécies Florestais com Deep Learning

Sistema de detecção de espécies florestais em imagens de drone usando Deep Forest (RetinaNet) + PyTorch Lightning. Construído do zero, sem IA-assisted coding — anotação manual de imagens, treino em GPU local, interpretação de curvas no TensorBoard. O projeto que originou toda a trajetória em Computação.

## O que faz?

Detecta e localiza espécies florestais (Cecropia Hololeuca, Fedegoso/Senna obtusifolia) em imagens aéreas de drone via bounding boxes. O pipeline cobre desde a conversão de anotações XML (LabelImg) para CSV, treino do modelo com Deep Forest, predição em novas imagens, até a validação de anotações. Evoluiu de uma versão CPU single-species (v1) para GPU multi-species (v2), e posteriormente para uma arquitetura refatorada (`model/`) com logging, early stopping e checkpoint automático.

## Funcionalidades

- Detecção de múltiplas espécies florestais por bounding box (Cecropia Hololeuca, Fedegoso)
- Pipeline de anotação: conversão XML (LabelImg) → CSV via `xmlpracsv2.py`
- Treino com Deep Forest 1.5.2 (RetinaNet backbone) + PyTorch Lightning
- Predição visual com matplotlib (bounding boxes + labels + scores)
- Validação de CSVs de anotação (labels, bounding boxes, integridade)
- Suporte CPU (v1) e GPU CUDA (v2) com early stopping e model checkpoint
- TensorBoard logging para monitoramento de treino

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Modelo | Deep Forest 1.5.2 (RetinaNet) |
| Framework | PyTorch + PyTorch Lightning |
| GPU | CUDA 11.8 / 12.x |
| Anotações | LabelImg (XML) → CSV |
| Visualização | matplotlib + OpenCV |
| Validação | pandas + scikit-learn |
| Ambiente | Conda (binturong.yaml) |

## Pré-requisitos

- Python 3.10+
- CUDA 11.8+ (para v2-gpu e model/)
- Conda (recomendado) ou pip

## Instalação

```bash
conda env create -f model/binturong.yaml
conda activate binturong
```

## Uso

### Treinar modelo (v2-gpu / model/)

```bash
cd model
python binturongfinal.py
```

### Predição em novas imagens

```bash
python model/prediz.py
```

### Validar anotações CSV

```bash
python model/valida.py
python model/validarcsv.py
```

### Converter anotações XML → CSV

```bash
python hololeuca/v2-gpu/annotations/xmlpracsv2.py
```

## Estrutura

```
forest-ai/
├── .gitignore
├── hololeuca/
│   ├── v1-cpu/                        # Versão 1: CPU, single-species (Cecropia Hololeuca)
│   │   ├── versions.py                # Info de versão PyTorch/CUDA
│   │   ├── trainer/
│   │   │   ├── pratatree.py           # Treinador principal (CPU)
│   │   │   ├── pratacecropia.py       # Treinador com albumentations + custom collate
│   │   │   ├── pratatreecallback.py   # Callbacks de treino
│   │   │   └── config.yml             # Config Deep Forest
│   │   ├── predict/
│   │   │   └── pratatreepredict.py    # Predição com visualização matplotlib
│   │   ├── annotations/               # CSVs de anotações + xmlpracsv2.py
│   │   └── saves/                     # Checkpoints salvos + XMLs originais
│   └── v2-gpu/                        # Versão 2: GPU, multi-species (Cecropia + Fedegoso)
│       ├── versions.py                # Info de versão GPU/CUDA
│       ├── trainer/
│       │   ├── gpufedegoso.py         # Treinador GPU para Fedegoso
│       │   ├── gpupratatreecallback.py # Callbacks GPU
│       │   └── config.yml             # Config Deep Forest
│       ├── predict/
│       │   └── gpupratatreepredict.py # Predição GPU com visualização
│       ├── annotations/               # CSVs de anotações (inclui bagunca/ para debug)
│       └── saves/                     # CSVs + scripts auxiliares
└── model/                             # Versão refatorada (Binturong): multi-species, GPU, prod-like
    ├── binturong.yaml                 # Conda environment
    ├── binturongfinal.py              # Trainer refatorado com logging, early stopping, checkpoint
    ├── binturong.py / binturong2.py   # Versões intermediárias
    ├── prediz.py                      # Predição com OpenCV (suporta paths não-ASCII)
    ├── valida.py                      # Validação completa de CSVs (labels, bboxes, imagens)
    ├── validarcsv.py                  # Correção automática de colunas Tree → label
    ├── renametree.py                  # Renomeador de colunas de label
    ├── xmlpracsv2.py                  # Conversor XML (LabelImg) → CSV
    ├── script.py                      # Script auxiliar
    ├── labels.csv                     # Anotações master
    ├── checkpoints/                   # Model checkpoints (gitignored)
    ├── fotos/                         # Imagens de treino (gitignored)
    ├── imagens-referencia/            # Imagens de referência por espécie
    └── testing/                       # Imagens de teste pós-treino
```

## Arquitetura

```
┌──────────────────────────────────────────────────┐
│ Imagens de Drone (JPG)                            │
│ + Anotações XML (LabelImg)                        │
├──────────────────────────────────────────────────┤
│ xmlpracsv2.py                                     │
│ XML → CSV (filename, xmin, ymin, xmax, ymax, label) │
├──────────────────────────────────────────────────┤
│ valida.py / validarcsv.py                         │
│ Validação: colunas, labels, bboxes, existência    │
├──────────────────────────────────────────────────┤
│ Deep Forest (RetinaNet)                           │
│ ┌─────────────┐  ┌──────────────┐  ┌───────────┐ │
│ │ v1-cpu      │  │ v2-gpu       │  │ model/    │ │
│ │ 1 espécie   │  │ 2+ espécies  │  │ prod-like │ │
│ │ CPU only    │  │ CUDA GPU     │  │ refatorado│ │
│ └─────────────┘  └──────────────┘  └───────────┘ │
├──────────────────────────────────────────────────┤
│ PyTorch Lightning Trainer                         │
│ EarlyStopping · ModelCheckpoint · TensorBoard     │
├──────────────────────────────────────────────────┤
│ prediz.py                                         │
│ Predição + Visualização (bounding boxes + scores) │
└──────────────────────────────────────────────────┘
```

## Configuração

Os caminhos de dados estão hardcoded nos scripts (estilo notebook). Para adaptar:

1. Ajuste `image_dir` e `annotations_csv` no script de treino correspondente
2. O `label_dict` define as espécies e seus índices (ex: `{"Cecropia Hololeuca": 0, "Fedegoso": 1}`)
3. O `config.yml` em cada versão controla o scheduler do Deep Forest

Para o model/ refatorado, use `setup_paths(base_dir)` para configurar diretórios de forma centralizada.

## Testes

Não há testes automatizados. Use `valida.py` e `validarcsv.py` para validar integridade dos dados antes do treino.

## Troubleshooting

| Problema | Solução |
|---|---|
| CUDA out of memory | Reduza `batch_size` no script de treino (padrão: 4-8) |
| Labels não encontradas | Execute `validarcsv.py` para corrigir coluna `Tree` → `label` |
| Imagens não carregam (path não-ASCII) | Use `prediz.py` com `cv2.imdecode` ao invés de `plt.imread` |
| Checkpoint não carrega | Verifique se `label_dict` e `num_classes` batem com o treino original |
| XML não converte | Verifique formato LabelImg — `xmlpracsv2.py` espera estrutura padrão |

## Contribuindo

Projeto pessoal. Alterações internas apenas.

## Licença

Proprietário — uso pessoal e acadêmico.
