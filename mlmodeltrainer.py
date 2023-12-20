"""Helpper module to train and evaluate sklearn models"""
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import catboost as cat
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score


def ml_trainer(X_train, y_train, X_val, y_val, model, preprocessor, cv: int = None):
    """Train sklearn ML models with preprocessing pipelines.
    If cross validion is used, give integer to cv parameter."""
    model_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("clf", model)]
    )
    
    # Train model
    if cv:
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=0)
        scores_t = []
        scores_v = []
        for train_i, val_i in skf.split(X_train, y_train):
            X_train_fold, X_val_fold = X_train.iloc[train_i], X_train.iloc[val_i]
            y_train_fold, y_val_fold = y_train.iloc[train_i], y_train.iloc[val_i]
            
            if isinstance(model, cat.CatBoostClassifier):
                preprocessor.fit(X_train_fold)
                X_val_fold_trans = preprocessor.transform(X_val_fold)
                params = {'clf__eval_set': (X_val_fold_trans, y_val_fold)}
            else:
                params = {}

            model_pipeline.fit(
                X_train_fold,
                y_train_fold,
                **params
            )
            proba_train_fold = model_pipeline.predict_proba(X_train_fold)
            score_t_fold = roc_auc_score(y_train_fold, proba_train_fold[:, 1])
            scores_t.append(score_t_fold)
            
            
            proba_val_fold = model_pipeline.predict_proba(X_val_fold)
            score_v_fold = roc_auc_score(y_val_fold, proba_val_fold[:, 1])
            scores_v.append(score_v_fold)
        
        
        print(f"CV Train scores AUC: {scores_t}")
        print(f"Mean CV Train score AUC: {np.mean(scores_t):.4f}")
        print(f"CV Validation scores AUC: {scores_v}")
        print(f"Mean CV Validation score AUC: {np.mean(scores_v):.4f}")
            
    else:  
        if isinstance(model, cat.CatBoostClassifier):
            preprocessor.fit(X_train)
            X_val_trans = preprocessor.transform(X_val)
            params = {'clf__eval_set': (X_val_trans, y_val)}
        else:
            params = {}

        model_pipeline.fit(
            X_train,
            y_train,
            **params
        )
        proba_train = model_pipeline.predict_proba(X_train)
        score_t = roc_auc_score(y_train, proba_train[:, 1])
        print(f"Train score AUC: {score_t:.4f}")
        
    
    # Evaluate
    proba_val = model_pipeline.predict_proba(X_val)
    score_v = roc_auc_score(y_val, proba_val[:, 1])
    print(f"Validation score AUC: {score_v:.4f}")
    
    # Inference time
    start_time = time.time()
    temp = model_pipeline.predict_proba(X_val.iloc[[0],:])
    end_time = start_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time: {inference_time*1000:.8f} ms")
    
    return model_pipeline, score_v, inference_time

