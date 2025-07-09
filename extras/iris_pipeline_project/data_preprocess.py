import pandas as pd
from sklearn.utils import shuffle
import os

os.makedirs("data_folder", exist_ok=True)

columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]

df = pd.read_csv("iris.csv", header=None, names=columns)
df.dropna(inplace=True)
df = shuffle(df, random_state=42).reset_index(drop=True)

df.to_csv("data_folder/iris_train.csv", index=False)
inference_sample = df.drop(columns=["Species"]).iloc[[0]]
inference_sample.to_csv("data_folder/iris_test.csv", index=False)

print("Cleaned and saved iris_train.csv and iris_test.csv to the 'data_folder/'")
