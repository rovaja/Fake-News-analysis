"""Helper module for models postprocessing"""
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightning as L
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)
from transformers import DistilBertTokenizer
from functools import partial
from lime.lime_text import LimeTextExplainer


def metrics(y_true, y_pred, y_proba = [], label_names = [], plot: bool=False):
    """Binary classification model performance metrics"""
    
    if plot:
        if len(y_proba) > 0:
            plot_roc_curve(y_true, y_proba)
        plot_confusion_matrix(y_true, y_pred, label_names)
        
    if len(label_names) > 0:
        cfl_report = classification_report(
            y_true, y_pred, zero_division=0, digits=4, target_names=label_names
            )
    else:
        cfl_report = classification_report(
            y_true, y_pred, zero_division=0, digits=4,
            )
    print(cfl_report)
        

def plot_roc_curve(y_true, y_pred):
    """Plots the ROC for clf."""
    if len(y_true) != len(y_pred):
        raise ValueError("Shapes of y_true and y_pred do not match.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    ax.plot(fpr, tpr, lw=2, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.set_title(f"Receiver Operating Characteristic (ROC) Curve")

    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')

def plot_confusion_matrix(y_true, y_pred, labels: [str] = []):
    """Create heatmap of confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.heatmap(
        confusion_matrix(y_true, y_pred),
        square=True,
        annot=True,
        fmt="",
        cbar=False,
    )
    if len(labels)>0:
        ax.set_xticklabels(labels, rotation=0, ha='right')
        ax.set_yticklabels(labels, rotation=90)
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title("Confusion matrix")
    

def distilbert_predictor(model, texts: np.array):
    """Predictor function of DistilBert model for LIME"""
    model = model.eval()
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
    proba_list = []
    
    for text in texts:
        encoded = tokenizer.encode_plus(
                        text,                      
                        add_special_tokens = True,
                        max_length = 128,
                        padding = 'max_length',
                        return_attention_mask = True,
                        truncation=True,
                        return_tensors='pt',
                )
        
        with torch.no_grad():
            probability, attention, logit = model(encoded['input_ids'], encoded['attention_mask'])
            proba_list.append(probability.item())
    probas = np.array(proba_list)
    neg_proba = 1 - probas
    return np.column_stack((neg_proba, probas))

def show_lime_explainer(
    model: L.LightningModule, sample_text: str, prediction_function, class_names = None
) -> None:
    """Create LIME output for given data instance"""
    explainer = LimeTextExplainer(class_names=class_names)
    prediction_function = partial(prediction_function, model)
    exp = explainer.explain_instance(sample_text, prediction_function, num_features=20, num_samples=1000)
    exp.show_in_notebook(text=sample_text)