import pandas as pd
from sklearn.utils import shuffle
import os
import sys

if len(sys.argv) < 2:
    print("Usage: python datapreprocess.py <output_directory>")
    sys.exit(1)

output_dir = sys.argv[1]

os.makedirs(output_dir, exist_ok=True)

columns = ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth", "Species"]

df = pd.read_csv("iris.csv", header=None, names=columns)
df.dropna(inplace=True)
df = shuffle(df, random_state=42).reset_index(drop=True)

train_path = os.path.join(output_dir, "iris_train.csv")
test_path = os.path.join(output_dir, "iris_test.csv")

df.to_csv(train_path, index=False)
df.drop(columns=["Species"]).iloc[[0]].to_csv(test_path, index=False)

print(f"Cleaned and saved iris_train.csv and iris_test.csv to '{output_dir}/'")
