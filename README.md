# OOD Detection Metrics

Functions for computing metrics commonly used in the field of out-of-distribution (OOD) detection.

<div style="overflow: hidden; display: flex; justify-content:flex-start; gap:10px;">
<a href="https://github.com/tayden/ood-metrics/actions/workflows/tests.yml">
<img height="19px" alt="Tests" src="https://github.com/tayden/ood-metrics/actions/workflows/tests.yml/badge.svg" />
</a>

<a href="https://github.com/tayden/ood-metrics/blob/main/LICENSE">
    <img alt="License" src="https://anaconda.org/conda-forge/ood-metrics/badges/license.svg" height="20px" />
</a>

<a href="https://anaconda.org/conda-forge/ood-metrics">
    <img alt="Version" src="https://anaconda.org/conda-forge/ood-metrics/badges/version.svg" height="20px" />
</a>
</div>

## Installation

### With PIP

`pip install ood-metrics`

### With Conda

`conda install -c conda-forge ood-metrics`

## Metrics functions

### AUROC

Calculate and return the area under the ROC curve using unthresholded predictions on the data and a binary true label.

```python
from ood_metrics import auroc

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

assert auroc(scores, labels) == 0.75
```

### AUPR

Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true
label.

```python
from ood_metrics import aupr

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

assert aupr(scores, labels) == 0.25
```

### FPR @ 95% TPR

Return the FPR when TPR is at least 95%.

```python
from ood_metrics import fpr_at_95_tpr

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

assert fpr_at_95_tpr(scores, labels) == 0.25
```

### Detection Error

Return the misclassification probability when TPR is 95%.

```python
from ood_metrics import detection_error

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

assert detection_error(scores, labels) == 0.05
```

### Calculate all stats

Using predictions and labels, return a dictionary containing all novelty detection performance statistics.

```python
from ood_metrics import calc_metrics

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

assert calc_metrics(scores, labels) == {
    'fpr_at_95_tpr': 0.25,
    'detection_error': 0.05,
    'auroc': 0.75,
    'aupr_in': 0.25,
    'aupr_out': 0.94375
}
```

## Plotting functions

### Plot ROC

Plot an ROC curve based on unthresholded predictions and true binary labels.

```python

from ood_metrics import plot_roc

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

plot_roc(scores, labels)
# Generate Matplotlib AUROC plot
```

### Plot PR

Plot an Precision-Recall curve based on unthresholded predictions and true binary labels.

```python

from ood_metrics import plot_pr

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

plot_pr(scores, labels)
# Generate Matplotlib Precision-Recall plot
```

### Plot Barcode

Plot a visualization showing inliers and outliers sorted by their prediction of novelty.

```python

from ood_metrics import plot_barcode

labels = [0, 0, 0, 1, 0]
scores = [0.1, 0.3, 0.6, 0.9, 1.3]

plot_barcode(scores, labels)
# Shows visualization of sort order of labels occording to the scores.
```
