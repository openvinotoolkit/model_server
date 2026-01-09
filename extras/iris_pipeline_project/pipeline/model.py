import abc
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearnex import patch_sklearn, unpatch_sklearn


class ModelClass(abc.ABC):
    @abc.abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, params: dict):
        """Train the model with given data and hyperparameters."""
        pass

    @abc.abstractmethod
    def predict(self, X: np.ndarray):
        """Run inference on the given data."""
        pass


class LogisticRegressionTorch(ModelClass):
    def __init__(self):
        self.model = None
        self.device = torch.device("cpu")
        self.use_ipex = False

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict):
        self.use_ipex = bool(params.get("use_ipex", False))

        try:
            if self.use_ipex:
                import intel_extension_for_pytorch as ipex
            else:
                ipex = None
        except ImportError:
            print("[Torch LogisticRegression] IPEX not available, falling back to native PyTorch.")
            self.use_ipex = False
            ipex = None

        X_tensor = torch.from_numpy(X).float().to(self.device)
        y_tensor = torch.from_numpy(y).long().to(self.device)

        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        self.model = nn.Linear(input_dim, num_classes)

        criterion = nn.CrossEntropyLoss()
        lr = params.get("lr", 0.01)
        epochs = params.get("epochs", 100)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)

        if self.use_ipex and ipex is not None:
            self.model, optimizer = ipex.optimize(self.model, optimizer=optimizer, dtype=torch.float32)

        start = time.perf_counter()
        self.model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
        end = time.perf_counter()

        print(f"[Torch LogisticRegression] Training time (IPEX={self.use_ipex}): {end - start:.4f} sec")
        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model not trained")

        X_tensor = torch.from_numpy(X).float().to(self.device)
        start = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1).cpu().numpy()
        end = time.perf_counter()

        print(f"[Torch LogisticRegression] Inference time: {end - start:.4f} sec")
        return preds, probs.cpu().numpy()


class KMeansSkLearn(ModelClass):
    def __init__(self):
        self.model = None
        self.use_onedal = False

    def fit(self, X: np.ndarray, y: np.ndarray, params: dict):
        self.use_onedal = bool(params.get("use_onedal", False))

        if self.use_onedal:
            patch_sklearn()
        else:
            unpatch_sklearn()

        n_clusters = params.get("n_clusters", 3)
        self.model = KMeans(n_clusters=n_clusters, random_state=42)

        start = time.perf_counter()
        self.model.fit(X)
        end = time.perf_counter()

        print(f"[Sklearn KMeans] Training time (oneDAL={self.use_onedal}): {end - start:.4f} sec")
        return self

    def predict(self, X: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model not trained")

        start = time.perf_counter()
        labels = self.model.predict(X)
        centroids = self.model.cluster_centers_
        end = time.perf_counter()
        print(f"[Sklearn KMeans] Inference time (oneDAL={self.use_onedal}): {end - start:.4f} sec")
        return labels, centroids

    def get_inertia(self):
        if self.model is None:
            raise RuntimeError("Model not trained")
        return self.model.inertia_
