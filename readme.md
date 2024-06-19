# Hoeffding Option Tree
Python implementation of Hoeffding Option Tree for Online Learning.
- [x] River integration.

## Requirements Installation
```
git clone https://github.com/francescogrillea/HoeffdingOptionTree.git
cd HoeffdingOptionTree

pip install -r requirements.txt
```

## Usage in Python
```python
from hoeffding_option_tree import HoeffdingOptionTreeClassifier
from river.evaluate import progressive_val_score
from river.metrics import Accuracy

dataset = ...  # add dataset
metric = ... # specify metric
model = HoeffdingOptionTreeClassifier(tau=1.5, grace_period=3)

progressive_val_score(dataset=dataset,
                      model=model,
                      metric=metric)
```