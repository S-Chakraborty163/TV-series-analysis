from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import evaluate
import torch

metric = evaluate.load('accuracy')
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def get_class_weights(df):
    unique_classes = np.array(sorted(df['label'].unique()))
    
    weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_classes,
        y=df['label'].values
    )
    return torch.tensor(weights, dtype=torch.float)