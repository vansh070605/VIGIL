import numpy as np
from src.evaluation.metrics import calculate_auroc, calculate_pro

def test_calculate_auroc():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.4, 0.35, 0.8])
    auroc = calculate_auroc(y_true, y_score)
    # The pairs are: (0.1, 0.35), (0.1, 0.8), (0.4, 0.35), (0.4, 0.8)
    # Correct ranking: 0.1 (0), 0.35 (1), 0.4 (0), 0.8 (1)
    # 0.8 > 0.1, 0.4. 0.35 > 0.1. So 3 out of 4 correct pairs -> 0.75
    assert np.isclose(auroc, 0.75)

    # Test single class
    assert calculate_auroc(np.array([0, 0]), np.array([0.1, 0.2])) == 0.5

def test_calculate_pro():
    # Ground truth: 2 connected components
    masks = np.array([
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0]
        ]
    ])
    
    # Anomaly map
    anomaly_maps = np.array([
        [
            [0.1, 0.1, 0.1, 0.1],
            [0.1, 0.9, 0.1, 0.1], # Hits component 1 (1/2 = 50%)
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.9, 0.1, 0.1]  # Hits component 2 (2/2 = 100%)
        ]
    ])
    
    # Threshold at 0.5
    pro = calculate_pro(masks, anomaly_maps, threshold=0.5)
    
    # Average PRO: (0.5 + 1.0) / 2 = 0.75
    assert np.isclose(pro, 0.75)
