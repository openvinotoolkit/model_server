import glob
import os
import re
from urllib.parse import urlparse, urlunparse

from google.cloud import storage


def gs_list_content(path):
    parsed_path = urlparse(path)
    bucket_name = parsed_path.netloc
    model_directory = parsed_path.path[1:]
    gs_client = storage.Client()
    bucket = gs_client.get_bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_directory)
    contents_list = []
    for blob in blobs:
        contents_list.append(blob.name)
    return contents_list


def get_versions_path(model_directory):
    if model_directory[-1] != os.sep:
        model_directory += os.sep
    parsed_model_dir = urlparse(model_directory)
    if parsed_model_dir.scheme == '':
        return glob.glob("{}/*/".format(model_directory))
    elif parsed_model_dir.scheme == 'gs':
        content_list = gs_list_content(model_directory)
        pattern = re.compile(parsed_model_dir.path[1:-1] + '/\d+/$')
        version_dirs = list(filter(pattern.match, content_list))
        return [urlunparse((parsed_model_dir.scheme, parsed_model_dir.netloc,
                            version_dir, parsed_model_dir.params,
                            parsed_model_dir.query, parsed_model_dir.fragment))
                for version_dir in version_dirs]


def get_version_number(version_directory):
    version_number = re.search('/\d+/$', version_directory).group(0)[1:-1]
    return int(version_number)


def get_full_path_to_model(specific_version_model_path):
    parsed_version_path = urlparse(specific_version_model_path)
    if parsed_version_path.scheme == '':
        bin_path = glob.glob("{}*.bin".format(specific_version_model_path))
        xml_path = glob.glob("{}*.xml".format(specific_version_model_path))
        if xml_path[0].replace('xml', '') == bin_path[0].replace('bin', ''):
            return xml_path[0], bin_path[0]
        return None, None
    elif parsed_version_path.scheme == 'gs':
        content_list = gs_list_content(specific_version_model_path)
        xml_pattern = re.compile(parsed_version_path.path[1:-1] + '/\w+\.xml$')
        bin_pattern = re.compile(parsed_version_path.path[1:-1] + '/\w+\.bin$')
        xml_path = list(filter(xml_pattern.match, content_list))
        bin_path = list(filter(bin_pattern.match, content_list))
        if xml_path[0].replace('xml', '') == bin_path[0].replace('bin', ''):
            xml_path[0] = urlunparse(
                (parsed_version_path.scheme, parsed_version_path.netloc,
                 xml_path[0], parsed_version_path.params,
                 parsed_version_path.query, parsed_version_path.fragment))
            bin_path[0] = urlunparse(
                (parsed_version_path.scheme, parsed_version_path.netloc,
                 bin_path[0], parsed_version_path.params,
                 parsed_version_path.query, parsed_version_path.fragment))
            return xml_path[0], bin_path[0]
        return None, None
