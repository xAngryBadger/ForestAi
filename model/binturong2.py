import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from deepforest import main
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split

# --- Performance Optimizations for NVIDIA GPUs ---
# Enable Tensor Cores for a significant speedup on RTX cards
torch.set_float32_matmul_precision('high')
# Enable CUDNN auto-tuner to find the best algorithm for your hardware
torch.backends.cudnn.benchmark = True
# ------------------------------------------------

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)


def setup_paths(base_dir: str) -> Dict[str, Path]:
    """Create and return all necessary directory paths."""
    base_path = Path(base_dir)
    paths = {
        "base": base_path,
        "images": base_path / "fotos",
        "annotations_csv": base_path / "labels.csv",
        "checkpoints": base_path / "checkpoints",
    }
    paths["train_csv"] = paths["checkpoints"] / "train_annotations.csv"
    paths["val_csv"] = paths["checkpoints"] / "val_annotations.csv"

    paths["checkpoints"].mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoint directory ensured at: {paths['checkpoints']}")
    
    return paths


def load_and_prepare_annotations(
    annotations_csv: Path,
    image_dir: Path,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Load, process, and split annotations into training and validation sets."""
    
    if not annotations_csv.exists():
        raise FileNotFoundError(f"Annotations file not found at: {annotations_csv}")

    df = pd.read_csv(annotations_csv, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'Tree': 'label'}) # Or whatever the column name is
    df["image_path"] = df["filename"].apply(lambda x: str(image_dir / Path(x).name))
    logging.info(f"Loaded {len(df)} annotations. Columns: {df.columns.tolist()}")

    unique_labels = sorted(df["label"].unique())
    label_dict = {label: i for i, label in enumerate(unique_labels)}
    logging.info(f"Created 0-indexed label dictionary: {label_dict}")

    if not set(df["label"]).issubset(set(label_dict.keys())):
        unknown_labels = set(df["label"]) - set(label_dict.keys())
        raise ValueError(f"Found labels in CSV not present in label_dict: {unknown_labels}")

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
    """Configures and runs the PyTorch Lightning training loop."""
    train_df.to_csv(paths["train_csv"], index=False)
    val_df.to_csv(paths["val_csv"], index=False)
    logging.info(f"Train/validation CSVs saved to {paths['checkpoints']}")

    num_object_classes = len(label_dict)
    config_args = {"num_classes": num_object_classes}
    model = main.deepforest(config_args=config_args, label_dict=label_dict)
    
    total_model_classes = num_object_classes + 1
    logging.info(f"Model configured for {total_model_classes} classes (including background).")

    # Configure model parameters
    model.config["train"]["csv_file"] = str(paths["train_csv"])
    model.config["train"]["root_dir"] = str(paths["images"])
    model.config["train"]["epochs"] = epochs
    model.config["train"]["batch_size"] = batch_size
    model.config["train"]["lr"] = lr
    model.config["train"]["num_workers"] = num_workers
    model.config["train"]["preload_images"] = True
    # Let deepforest use its default scheduler monitor. It knows its own metrics best.
    
    model.config["validation"]["csv_file"] = str(paths["val_csv"])
    model.config["validation"]["root_dir"] = str(paths["images"])
    model.config["validation"]["batch_size"] = batch_size
    model.config["validation"]["num_workers"] = num_workers
    model.config["validation"]["score_thresh"] = 0.1
    
    # --- PATCHED: Monitor 'map' (mean average precision), the standard for object detection. ---
    # We want to MAXIMIZE mAP, so mode is 'max'.
    model_checkpoint = ModelCheckpoint(
        dirpath=str(paths["checkpoints"]),
        filename="best_model-{epoch:02d}-{map:.2f}",
        save_top_k=1,
        verbose=True,
        monitor="map",
        mode="max"
    )
    
    early_stopping = EarlyStopping(
        monitor="map",
        patience=10, 
        verbose=True,
        mode="max"
    )

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
        precision="16-mixed",
        callbacks=[model_checkpoint, early_stopping]
    )
    
    logging.info(f"Starting training for {epochs} epochs on '{accelerator}' with {num_workers} workers...")
    trainer.fit(model)
    logging.info("Training completed successfully.")
    
    if model_checkpoint.best_model_path:
        logging.info(f"The best model was saved to: {model_checkpoint.best_model_path}")
    else:
        logging.warning("Could not find the best model path. Checkpoint callback might not have saved a model.")


def main_workflow():
    """Main execution workflow to set up and run the training process."""
    
    # --- Main Configuration ---
    BASE_DIR = r"F:\CÓDIGO\model"
    EPOCHS = 60
    LEARNING_RATE = 0.0005
    BATCH_SIZE = 16  # Increased for modern GPU
    # Use 6 workers for an i5-13400F (matches P-core count)
    NUM_WORKERS = 6
    # --------------------------
    
    # --- System Detection ---
    use_gpu = torch.cuda.is_available()
    ACCELERATOR = "gpu" if use_gpu else "cpu"
    DEVICES = 1 if use_gpu else "auto" # 'auto' for CPU to let lightning decide
    
    if not use_gpu:
        logging.warning("CUDA is not available. Training will run on CPU, which may be slow.")
        logging.warning("If you have an NVIDIA GPU, please check your PyTorch and CUDA installation.")
    # ------------------------

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
    main_workflow()