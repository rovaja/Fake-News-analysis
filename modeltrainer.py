"""Helpper module to train and evaluate pytorch lightning models"""
import torch
import time
from torch.utils.data import DataLoader, Subset
import lightning as L
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score


def train(
    model: L.LightningModule,
    model_name: str,
    version: str,
    max_epochs: int = 1,
    device: str = "gpu",
    lr: float = None,
    log_freq: int = 200
) -> None:
    """Train pytorch model, save the best version and plot metrics.

    Give learning rate if the automatic search is not used."""

    print(f"---Currently {model_name} version {version} in training.---")
    model = model.train()
    early_stop_callback = L.pytorch.callbacks.early_stopping.EarlyStopping(
        monitor="val_loss", mode="min", patience=2
    )
    csv_logger = L.pytorch.loggers.CSVLogger(
        save_dir="logs", name=model_name, version=version, flush_logs_every_n_steps = int(log_freq*2)
    )

    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath=f"logs/{model_name}/{version}/best_ckpt",
        filename=model_name + "_epoch{epoch:02d}-val_loss{val_loss:.2f}",
        auto_insert_metric_name=False,
        monitor="val_loss",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        max_time="00:04:00:00",
        accelerator=device,
        logger=csv_logger,
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        log_every_n_steps=log_freq,
    )
    torch.set_float32_matmul_precision("medium")

    if not lr:
        tuner = L.pytorch.tuner.Tuner(trainer)
        tuner.lr_find(model, max_lr=0.005)
    else:
        model.lr == lr
    print(f"---Selected starting learning rate: {model.lr}---")

    trainer.fit(model)

    del model
    torch.cuda.empty_cache()


def plot_metrics(
    model_name: str, version: str, metrics: list[str], titles: list[str]
) -> None:
    """Plot metrics saved in csv-file by Lightning trainer.
    Give column names without train/val prefix."""
    path = f"logs/{model_name}/{version}/metrics.csv"
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 5))
    axes = axes.flatten()
    for ax, metric, title in zip(axes, metrics, titles):
        train_label = f"train_{metric}"
        val_label = f"val_{metric}"
        df[["epoch", train_label]].dropna().plot.line(
            x="epoch", y=train_label, label=f"train_{title}", ax=ax
        )
        df[["epoch", val_label]].dropna().plot.line(
            x="epoch", y=val_label, label=f"val_{title}", ax=ax
        )
        ax.legend().set_visible(True)
        ax.set_title(title)
        ax.set_xlabel("epoch")
    fig.suptitle(f"Model: {model_name}", y=0.98)


def evaluate(
    model: L.LightningModule,
    model_name: str,
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> dict:
    """Evalute model against given dataset.
    Inference time is evaluted using CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.eval()
    model = model.to(device)

    true_dict = defaultdict(list)
    predicted_dict = defaultdict(list)

    with torch.no_grad():
        for i, batch in enumerate(DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )):
            ids = [j + i*batch_size  for j in range(batch_size)]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["target"].to(device)
            proba, _ , logit = model(input_ids, attention_mask)

            predicted_dict["label"].extend(
                (proba.squeeze() > 0.5).int().cpu().numpy()
            )
            predicted_dict["proba"].extend(proba.squeeze().cpu().numpy())
            predicted_dict["ids"].extend(ids)

            true_dict["label"].extend(labels.int().cpu().numpy())
            true_dict["attention_mask"].extend(attention_mask.cpu().numpy())
            true_dict["input_ids"].extend(input_ids.cpu().numpy())

    # Separate loop to estimate the inference time with CPU
    inference_times = []
    device = "cpu"
    model = model.to(device)
    mini_dataset = Subset(dataset, [1, 2, 3, 4, 5])
    with torch.no_grad():
        for batch in DataLoader(
            mini_dataset, batch_size=1, shuffle=False, num_workers=num_workers
        ):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_time = time.time()
            outputs = model(input_ids, attention_mask)
            end_time = time.time()
            inference_times.append(end_time - start_time)

    auc_score = roc_auc_score(true_dict["label"], predicted_dict["proba"])
    avg_inference_time = np.mean(inference_times)
    metrics_df = pd.DataFrame(
        {
            "model": model_name,
            "ROC AUC": auc_score,
            "Average Inference Time": avg_inference_time,
        },
        index=[0],
    )
    metrics_df = metrics_df.set_index("model")

    return {
        "metrics": metrics_df,
        "predictions": predicted_dict,
        "true_labels": true_dict,
    }