from urllib.parse import urlparse

from ie_serving.models.gs_model import GSModel
from ie_serving.models.local_model import LocalModel
from ie_serving.models.s3_model import S3Model


class ModelBuilder:
    @staticmethod
    def build(model_name: str, model_directory: str, batch_size):
        parsed_path = urlparse(model_directory)
        if parsed_path.scheme == '':
            return LocalModel.build(model_name, model_directory, batch_size)
        elif parsed_path.scheme == 'gs':
            return GSModel.build(model_name, model_directory, batch_size)
        elif parsed_path.scheme == 's3':
            return S3Model.build(model_name, model_directory, batch_size)
