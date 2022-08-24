from .metrics import aupr, auroc, detection_error, fpr_at_95_tpr


def test_auroc():
    assert auroc([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    assert auroc([0.4, 0.3, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auroc([0.4, 0.3, 0.2, 0.1], [0, 1, 1, 0]) == 0.5
    assert auroc([0.4, 0.3, 0.2, 0.1], [-1, 1, 1, -1]) == 0.5
    assert auroc([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0]) == 0.0
    assert auroc([0.1, 0.2, 0.3, 0.4], [1, 0, 1, 1]) == 2. / 3


def test_aupr():
    assert aupr([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 1.0
    assert round(aupr(list(range(10000)), [i % 2 for i in range(10000)]), 2) == 0.5


def test_fpr_at_95_tpr():
    assert fpr_at_95_tpr([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 0.0
    assert fpr_at_95_tpr([0.1, 0.2, 0.3, 0.4], [1, 1, 0, 0]) == 1.0
    assert round(fpr_at_95_tpr(list(range(10000)), [i % 2 for i in range(10000)]), 2) == 0.95


def test_detection_error():
    assert detection_error([0.1, 0.2, 0.3, 0.4], [0, 0, 1, 1]) == 0.0
    assert round(detection_error(list(range(100)), [1] * 3 + [0] * 97), 2) == 0.03
    assert round(detection_error(list(range(100)), [1] * 4 + [0] * 96), 2) == 0.04
    assert round(detection_error(list(range(10000)), [i % 2 for i in range(10000)]), 2) == 0.5
