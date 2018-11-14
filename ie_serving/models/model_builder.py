from urllib.parse import urlparse

from ie_serving.models.gs_model import GSModel
from ie_serving.models.local_model import LocalModel


class ModelBuilder:
    @staticmethod
    def build(model_name: str, model_directory: str):
        parsed_path = urlparse(model_directory)
        if parsed_path.scheme == '':
            return LocalModel.build(model_name, model_directory)
        elif parsed_path.scheme == 'gs':
            return GSModel.build(model_name, model_directory)
