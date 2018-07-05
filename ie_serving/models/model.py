import glob
import os
from ie_serving.models.ir_engine import IrEngine

class Model():

    def __init__(self, model_name: str, model_directory: str):
        self.model_name = model_name
        self.model_directory = model_directory
        self.versions = self.get_all_available_versions()
        self.engines = self.get_engines_for_model()
        self.versions = [version['version'] for version in self.versions]
        self.default_version = max(self.versions)

    def get_absolut_path_to_model(self, specific_version_model_path):
        bin_path = glob.glob("{}/*.bin".format(specific_version_model_path))
        xml_path = glob.glob("{}/*.xml".format(specific_version_model_path))
        if xml_path[0].replace('xml', '') == bin_path[0].replace('bin', ''):
            return xml_path[0], bin_path[0]
        return None, None

    def get_all_available_versions(self):
        versions_path = glob.glob("{}/*/".format(self.model_directory))
        versions = []
        for version in versions_path:
            number = self.get_version_number_of_model(version_path=version)
            if number != 0:
                model_xml, model_bin = self.get_absolut_path_to_model(os.path.join(self.model_directory, version))
                if model_xml is not None and model_bin is not None:
                    model_info = {'xml_model_path': model_xml, 'bin_model_path': model_bin, 'version': number}
                    versions.append(model_info)
        return versions

    def get_version_number_of_model(self, version_path):
        folder_name = os.path.basename(os.path.normpath(version_path))
        try:
            number_version = int(folder_name)
            return number_version
        except ValueError:
            return 0

    def get_engines_for_model(self):
        inference_engines = {}
        for version in self.versions:
            inference_engines[version['version']] = IrEngine(model_bin=version['bin_model_path'],
                                                             model_xml=version['xml_model_path'])
        return inference_engines
