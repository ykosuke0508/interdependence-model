# interdependence-model
====

# Overview
These codes were used in the experiments of our study, ``Multi-label Classification by Interdependence Model.``
In this study, we proposed ``interdependence model`` for multi-label classification. This method can clearly deal with relationship between labels to predict the labels.

## Demo (normal multi-label setting)

```
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from evaluation_metric import metrics
from proposed_method import interdependence_model
from download_datasets.multilabel_datasets import load_scene
from conventional_method import problem_transformation_methods as ptm

# load scene dataset if you have; otherwise download scene zip, unzip the file, and load the data.
data = load_scene()

# feature data
X = data.features

# label data
y = data.labels

# make train data and test data
n_samples = data.n_samples
test_ratio = 0.1
n_test_samples = int(n_samples * test_ratio)

#NOTE: in this demo, you didn't shuffle data.
X_test = X[:n_test_samples]
X_train = X[n_test_samples:]

y_test = y[:n_test_samples]
y_train = y[n_test_samples:]

# Normalize Data X
ms = MinMaxScaler()

X_train = ms.fit_transform(X_train)
X_test = ms.transform(X_test)

# proposed model
IDD = interdependence_model.InterdependenceModel(LogisticRegression(C = 1), prediction_method = "ESwP")

# conventional model
BR = ptm.BinaryRelevance(LogisticRegression(C = 1))

# training step
IDD.fit(X_train, y_train)
BR.fit(X_train, y_train)

# prediction step
IDD_pred = IDD.predict(X_test)
BR_pred = BR.predict(X_test)

# evaluation
IDD_score = metrics.accuracy(y_test, IDD_pred)
BR_score = metrics.accuracy(y_test, BR_pred)

# output results
print("Accuracy of IDD model :  " + str(IDD_score))
print("Accuracy of BR model :  " + str(BR_score))

IDD.print_time()
```

## Licence

[MIT](https://github.com/ykosuke0508/interdependence-model/blob/master/LICENSE)

## Author

[ykosuke0508](https://github.com/ykosuke0508)
