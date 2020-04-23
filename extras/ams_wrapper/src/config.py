from api.models.example_model import ExampleModel
from api.ovms_connector import OvmsConnector
OVMS_PORT = 9000

# Default version for a model is the latest one
DEFAULT_VERSION = -1 
AVAILABLE_MODELS = [
    {"name": "model", 
    "class": ExampleModel, 
    "ovms_mapping": {
        "model_name": "model",
        "model_version": DEFAULT_VERSION,
        "input_name": "input",
        "input_shape": (1, 3, 200, 200), 
        }
    },
]
