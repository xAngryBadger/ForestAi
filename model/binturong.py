# -*- coding: utf-8 -*-
"""
Script to train a multi-species DeepForest object detection model.

This script performs the following steps:
1.  Sets up project directories using pathlib.
2.  Loads annotations from a CSV file.
3.  Preprocesses the data: cleans column names, remaps labels for DeepForest (1-based indexing).
4.  Splits the data into training and validation sets, stratifying by label.
5.  Configures a DeepForest model and a PyTorch Lightning Trainer.
6.  Trains the model.
7.  Saves the final trained model checkpoint.
"""
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from deepforest import main
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def setup_paths(base_dir: str) -> Dict[str, Path]:
    """
    Defines and creates all necessary directory and file paths.

    Args:
        base_dir (str): The root directory for the project.

    Returns:
        Dict[str, Path]: A dictionary of important paths.
    """
    base_path = Path(base_dir)
    paths = {
        "base": base_path,
        "images": base_path / "fotos",
        "annotations_csv": base_path / "labels.csv",
        "checkpoints": base_path / "checkpoints",
    }
    paths["train_csv"] = paths["checkpoints"] / "train_annotations.csv"
    paths["val_csv"] = paths["checkpoints"] / "val_annotations.csv"
    paths["saved_model"] = paths["checkpoints"] / "deepforest_multispecies.pt"

    # Create the checkpoint directory if it doesn't exist
    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoint directory ensured at: {paths['checkpoints']}")
    
    return paths


def load_and_prepare_annotations(
    annotations_csv: Path,
    image_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Loads, cleans, and splits the annotations data.

    Args:
        annotations_csv (Path): Path to the raw annotations CSV file.
        image_dir (Path): Path to the directory containing images.
        test_size (float): The proportion of the dataset to allocate to the validation set.
        random_state (int): Seed for the random number generator for reproducibility.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]: A tuple containing:
            - The training DataFrame.
            - The validation DataFrame.
            - The label dictionary mapping class names to integers.
    """
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations file not found at: {annotations_csv}")

    # 1. Carregar e limpar dados (Load and clean data)
    df = pd.read_csv(annotations_csv, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Tree': 'label'})
    df["image_path"] = df["filename"].apply(lambda x: str(image_dir / Path(x).name))
    logging.info(f"Loaded {len(df)} annotations. Columns: {df.columns.tolist()}")

    # 2. Criar label_dict (Create label dictionary)
    # DeepForest requires object labels to start from 1 (0 is reserved for background)
    unique_labels = sorted(df["label"].unique())
    label_dict = {label: i + 1 for i, label in enumerate(unique_labels)}
    logging.info(f"Created label dictionary: {label_dict}")

    # 3. Validar rótulos (Validate labels)
    if not set(df["label"]).issubset(set(label_dict.keys())):
        unknown_labels = set(df["label"]) - set(label_dict.keys())
        raise ValueError(f"Found labels in CSV not present in label_dict: {unknown_labels}")

    # 4. Dividir em treino/validação (Split into train/validation)
    required_cols = ["image_path", "xmin", "ymin", "xmax", "ymax", "label"]
    train_df, val_df = train_test_split(
        df[required_cols],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"]
    )
    logging.info(f"Data split into {len(train_df)} training and {len(val_df)} validation samples.")
    
    return train_df, val_df, label_dict


def train_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_dict: Dict[str, int],
    paths: Dict[str, Path],
    epochs: int,
    batch_size: int,
    lr: float,
    num_workers: int,
    accelerator: str = "cpu",
    devices: int = 1
):
    """
    Configures and trains the DeepForest model.

    Args:
        train_df (pd.DataFrame): The training data.
        val_df (pd.DataFrame): The validation data.
        label_dict (Dict[str, int]): The label-to-integer mapping.
        paths (Dict[str, Path]): Dictionary of project paths.
        epochs (int): Number of training epochs.
        batch_size (int): Number of samples per batch.
        lr (float): Learning rate.
        num_workers (int): Number of workers for the DataLoader.
        accelerator (str): Hardware accelerator to use ('cpu', 'gpu', 'tpu').
        devices (int): Number of devices to use.
    """
    # Save the split dataframes to CSV for DeepForest to consume
    train_df.to_csv(paths["train_csv"], index=False)
    val_df.to_csv(paths["val_csv"], index=False)
    logging.info(f"Train/validation CSVs saved to {paths['checkpoints']}")

    # 1. Configurar modelo (Configure model)
    # Total classes = number of object classes + 1 for the background class
    num_classes = len(label_dict) + 1
    
    model = main.deepforest()
    model.config["train"]["csv_file"] = str(paths["train_csv"])
    model.config["train"]["root_dir"] = str(paths["images"])
    model.config["train"]["epochs"] = epochs
    model.config["train"]["batch_size"] = batch_size
    model.config["train"]["lr"] = lr
    model.config["train"]["num_workers"] = num_workers
    # Use preload_images=True with GPU for faster training if memory allows
    model.config["train"]["preload_images"] = (accelerator == "gpu")
    
    model.config["validation"]["csv_file"] = str(paths["val_csv"])
    model.config["validation"]["root_dir"] = str(paths["images"])
    model.config["validation"]["batch_size"] = batch_size
    model.config["validation"]["num_workers"] = num_workers

    # Critical: Set the number of classes and the label dictionary
    model.config["num_classes"] = num_classes
    model.label_dict = label_dict
    
    logging.info(f"Model configured for {num_classes} classes (including background).")

    # 2. Configurar Trainer (Configure Trainer)
    logger = TensorBoardLogger(
        save_dir=str(paths["checkpoints"]), 
        name="training_logs"
    )
    trainer = pl.Trainer(
        max_epochs=epochs,
        logger=logger,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
    )
    
    # 3. Treinar o modelo (Train the model)
    logging.info(f"Starting training for {epochs} epochs on '{accelerator}'...")
    trainer.fit(model)
    logging.info("Training completed successfully.")
    
    # 4. Salvar modelo final (Save final model)
    # trainer.save_checkpoint(paths["saved_model"]) # PyTorch Lightning format
    model.save_model(str(paths["saved_model"])) # Recommended DeepForest format
    logging.info(f"Model saved to: {paths['saved_model']}")


def main_workflow():
    """Main execution workflow."""
    
    # --- Configuration ---
    # Altere estas variáveis conforme necessário (Change these variables as needed)
    BASE_DIR = r"E:\deepbug ai\model"
    EPOCHS = 120
    BATCH_SIZE = 4
    LEARNING_RATE = 0.0005
    # Defina o número de workers com base nos núcleos da sua CPU (Set num_workers based on your CPU cores)
    NUM_WORKERS = 2 if os.name == 'nt' else 4 # Good default for Windows vs. Linux/Mac
    # Mude para 'gpu' para treinar na placa de vídeo (Change to 'gpu' to train on a graphics card)
    ACCELERATOR = "cpu"  
    DEVICES = 1 # Number of GPUs/CPUs to use

    # --- Execution ---
    paths = setup_paths(BASE_DIR)

    train_df, val_df, label_dict = load_and_prepare_annotations(
        annotations_csv=paths["annotations_csv"],
        image_dir=paths["images"],
        test_size=0.2,
        random_state=42
    )

    train_model(
        train_df=train_df,
        val_df=val_df,
        label_dict=label_dict,
        paths=paths,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        lr=LEARNING_RATE,
        num_workers=NUM_WORKERS,
        accelerator=ACCELERATOR,
        devices=DEVICES
    )
    
if __name__ == "__main__":
    # Check for PyTorch GPU availability and provide a hint
    if not torch.cuda.is_available():
        logging.warning("CUDA is not available. Training will run on CPU, which may be slow.")
        logging.warning("If you have an NVIDIA GPU, please check your PyTorch and CUDA installation.")
        
    main_workflow()