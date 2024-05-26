from river.tree import HoeffdingTreeClassifier
from hoeffding_option_tree import HoeffdingOptionTreeClassifier
from river.stream import iter_sklearn_dataset
from sklearn import datasets
from river.metrics import Accuracy

model = HoeffdingOptionTreeClassifier(tau=1.5, grace_period=3)
# model = HoeffdingTreeClassifier(tau=1.5, grace_period=3)
dataset = datasets.load_iris()
metric = Accuracy()
plot_metric = []

for i, (x, y) in enumerate(iter_sklearn_dataset(dataset, shuffle=True, seed=42)):
    y_pred = model.predict_one(x)
    if y_pred is not None:
        metric.update(y, y_pred)
        plot_metric.append(metric.get())
    model.learn_one(x, y)


print(metric)
print(metric.cm)