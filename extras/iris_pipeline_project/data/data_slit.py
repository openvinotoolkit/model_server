import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data/Iris (copy).csv') 

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

train_df.to_csv('data/iris_train.csv', index=False)
test_df.to_csv('data/iris_test.csv', index=False)