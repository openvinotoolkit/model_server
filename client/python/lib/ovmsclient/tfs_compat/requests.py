from abc import ABC

class PredictRequest(ABC):
    
    def __init__(self, inputs, model_name, model_version):
        self.inputs = inputs
        self.model_name = model_name
        self.model_version = model_version

class ModelMetadataRequest(ABC):

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version

class ModelStatusRequest(ABC):

    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version