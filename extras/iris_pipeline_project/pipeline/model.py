import abc
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans


class ModelClass(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, params: dict):
        """
        Train the model and return a serializable trained object (could be self or something to export).
        """
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        """
        Return predictions for X.
        """
        pass


class LogisticRegressionTorch(ModelClass):
    def __init__(self):
        self.model = None
        self.device = torch.device("cpu")

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict):
        try:
            import intel_extension_for_pytorch as ipex  
            use_ipex = True
        except ImportError:
            use_ipex = False

        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)

        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = nn.Linear(input_dim, num_classes)

        criterion = nn.CrossEntropyLoss()
        lr = params.get("lr", 0.01)
        epochs = params.get("epochs", 100)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        if use_ipex:
            self.model, optimizer = ipex.optimize(self.model, optimizer=optimizer, dtype=torch.float32)

        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        return self  

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model not trained")
        self.model.eval()
        X_tensor = torch.from_numpy(X).float()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        return preds, probs.cpu().numpy()

class KMeansSkLearn(ModelClass):
    def __init__(self):
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict):
        n_clusters = params.get("n_clusters", 3)
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.predict(X)
