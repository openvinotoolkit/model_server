from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import os
import openvino as ov

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

joblib.dump(model, 'logreg_model.pkl')

from skl2onnx import to_onnx

onx = to_onnx(model, X_train[:1], target_opset=12)
ov.convert_model('logreg_model.onnx')

import onnx
model = onnx.load("model/iris_logreg/1/logreg_model.onnx")
print("Inputs:", [i.name for i in model.graph.input])
print("Outputs:", [o.name for o in model.graph.output])
