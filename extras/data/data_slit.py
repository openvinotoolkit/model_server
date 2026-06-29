import pandas as pd
from sklearn.model_selection import train_test_split

# Load the complete dataset
df = pd.read_csv('data/Iris (copy).csv')  # Update this path as needed

# Split the data (80% train, 20% test by default)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

# Save to new CSV files
train_df.to_csv('data/iris_train.csv', index=False)
test_df.to_csv('data/iris_test.csv', index=False)