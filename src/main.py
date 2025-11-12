import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_split_mnist(train_frac: float = 0.6, random_state: int = 42):
  
  df = pd.read_csv('data/mnist_data.csv', header=None)
  X, y = df.iloc[:, 1:].values, df.iloc[:, 0].values

  # Split while preserving class proportions
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=train_frac, stratify=y, random_state=random_state
  )
  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
  X_train, X_test, y_train, y_test = load_and_split_mnist()
  print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
  print(f"X_test  shape: {X_test.shape}, y_test  shape: {y_test.shape}")