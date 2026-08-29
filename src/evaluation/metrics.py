import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auroc(y_true, y_score):
    """
    Calculate the Area Under the Receiver Operating Characteristic Curve (AUROC).
    
    Args:
        y_true (np.array): Ground truth binary labels.
        y_score (np.array): Predicted scores (higher means more anomalous).
        
    Returns:
        float: AUROC score
    """
    if len(np.unique(y_true)) == 1:
        # If there's only one class present (e.g. all normal), AUROC is undefined.
        return 0.5
        
    return roc_auc_score(y_true, y_score)
