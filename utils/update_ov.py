#
# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# This script is pretty much a draft that has quite chaotic flow and probably a few unnecessary things.
# It was created to automate OV dependencies update process and if it's supposed to be merged then it will require some improvements for
# better readability and maintainability.

# This script is intended to be run from the root of the OpenVINO Model Server repository.
#   - It is supposed to: update the Makefile and other files in the OpenVINO repository
#     with the latest OpenVINO and OpenVINO GenAI package links and commit SHAs.
#   - The script uses Selenium WebDriver to fetch the latest package links from the OpenVINO
#     and OpenVINO GenAI storage. 
#   - It then extracts the commit SHAs and dates from the links
#     and updates the Makefile with the new package links and commit SHAs.
#   - The script updates the requirements.txt file for the export models script with the new
#     OpenVINO and OpenVINO Tokenizers versions.
#   - The script updates the llm_engine.bzl file with the new OpenVINO GenAI commit SHA.
#   - The script updates the windows_install_build_dependencies.bat file with the new OpenVINO GenAI package link.

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import re

### OpenVINO Part ###
def update_openvino():
    print("Updating OpenVINO")
    # URL to fetch the latest package links
    openvino_url = "https://storage.openvinotoolkit.org/repositories/openvino/packages/nightly/latest/"

    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Fetch the page content
    driver.get(openvino_url)
    # Wait for the page to fully load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
    response_content = driver.page_source
    driver.quit()

    openvino_packages_content = response_content
    soup = BeautifulSoup(openvino_packages_content, 'html.parser')

    # Extract all links
    all_links = [a['href'] for a in soup.find_all('a', href=True)]
    # Filter package links to contain only links to tgz or zip files
    package_links = [link for link in all_links if link.endswith('.tgz') or link.endswith('.zip')]
    print(f"Available OpenVINO packages: {package_links}")

    # Pick Ubuntu24 package link for extracting metadata
    u24_package_link = next(link for link in package_links if "ubuntu24" in link)

    # Extract the date from the link (assuming nighly package links contain a date in the format devYYYYMMDD)
    date_match = re.search(r'dev(\d{8})', u24_package_link)
    if date_match:
        date_str = date_match.group(1)
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        print(f"Extracted date: {date_formatted}")
    else:
        raise Exception("Date not found in the package link.")

    # Find a link that starts with the specified URL
    commit_link = next((link for link in all_links if link.startswith("https://github.com/openvinotoolkit/openvino/commit/")), None)

    # Extract the commit SHA from the link
    if commit_link:
        commit = commit_link.split("https://github.com/openvinotoolkit/openvino/commit/")[1]
        print(f"OpenVINO commit: {commit}")
    else:
        raise Exception("No link with the commit SHA found")

    # Read the Makefile
    makefile_path = 'Makefile'
    with open(makefile_path, 'r') as file:
        makefile_content = file.readlines()

    # OS identifiers
    os_identifiers = ['ubuntu24', 'ubuntu22', 'rhel8']

    # Update Makefile content with package links and OV commit
    new_makefile_content = []
    for line in makefile_content:
        if "OV_SOURCE_BRANCH ?=" in line:
            print("Updating OV_SOURCE_BRANCH")
            indentation = line[:line.index("OV_SOURCE_BRANCH ?=")]
            new_line = f'{indentation}OV_SOURCE_BRANCH ?= {commit} # master {date_formatted}\n'
            print(new_line.lstrip())
            new_makefile_content.append(new_line)
        elif "DLDT_PACKAGE_URL ?= " in line:
            indentation = line[:line.index("DLDT_PACKAGE_URL ?= ")]
            for os_id in os_identifiers:
                if os_id in line:
                    print("Updating DLDT_PACKAGE_URL for", os_id)
                    for link in package_links:
                        if os_id in link:
                            new_line = f'{indentation}DLDT_PACKAGE_URL ?= {link}\n'
                            print(new_line.lstrip())
                            new_makefile_content.append(new_line)
                            break
                    else:
                        new_makefile_content.append(line)
                    break
            else:
                new_makefile_content.append(line)
        else:
            new_makefile_content.append(line)
    
    with open(makefile_path, 'w') as file:
        file.writelines(new_makefile_content)

    # Update requirements for export models script
    model_export_requirements_path = "demos/common/export_models/requirements.txt"
    # Extract the package version from the package link in format <release_version>.dev<date>
    version_match = re.search(r'openvino_toolkit_[^_]+_(\d+\.\d+\.\d+\.dev\d+)', u24_package_link)
    if version_match:
        package_version = version_match.group(1)
        print(f"Extracted package version: {package_version}")
    else:
        raise Exception(f"No package version found in the package link: {u24_package_link}")
    
    # Update OpenVINO package version in requirements.txt
    with open(model_export_requirements_path, 'r') as file:
        model_export_requirements_content = file.readlines()
    new_model_export_requirements_content = []
    for line in model_export_requirements_content:
        if "openvino<=" in line:
            print("Updating openvino in requirements.txt")
            new_line = f"openvino<={package_version}\n"
            print(new_line)
            new_model_export_requirements_content.append(new_line)
        else:
            new_model_export_requirements_content.append(line)

    with open(model_export_requirements_path, 'w') as file:
        file.writelines(new_model_export_requirements_content)

def update_openvino_genai():
    print("Updating OpenVINO GenAI")
    ### OpenVINO GenAI Part ###
    openvino_genai_url = "https://storage.openvinotoolkit.org/repositories/openvino_genai/packages/nightly/latest/"
    
    # Set up Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)

    # Fetch the page content
    driver.get(openvino_genai_url)
    # Wait for the page to fully load
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "a")))
    response_content = driver.page_source
    driver.quit()

    openvino_genai_packages_content = response_content
    soup = BeautifulSoup(openvino_genai_packages_content, 'html.parser')

    # Extract all links
    all_links = [a['href'] for a in soup.find_all('a', href=True)]
    # Filter package links to contain only links to tgz or zip files
    package_links = [link for link in all_links if link.endswith('.tgz') or link.endswith('.zip')]
    print(f"Available OpenVINO GenAI packages: {package_links}")

    # Pick Windows package link for extracting metadata
    windows_package_link = next(link for link in package_links if "windows" in link)

    # Extract the date from the link (assuming nighly package links contain a date in the format devYYYYMMDD)
    date_match = re.search(r'dev(\d{8})', windows_package_link)
    if date_match:
        date_str = date_match.group(1)
        date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
        print(f"Extracted date: {date_formatted}")
    else:
        raise Exception("No date found in the package link.")

    # Find a link that starts with the specified URL
    commit_link = next((link for link in all_links if link.startswith("https://github.com/openvinotoolkit/openvino.genai/commit/")), None)
    # Extract commit SHA from the link
    if commit_link:
        commit = commit_link.split("https://github.com/openvinotoolkit/openvino.genai/commit/")[1]
        print(f"OpenVINO GenAI commit: {commit}")
    else:
        raise Exception("No commit link found.")

    # Update OpenVINO GenAI commit in llm_engine.bzl
    llm_engine_bzl_path = "third_party/llm_engine/llm_engine.bzl"
    with open(llm_engine_bzl_path, 'r') as file:
        llm_engine_bzl_content = file.readlines()
    new_llm_engine_bzl_content = []
    for line in llm_engine_bzl_content:
        if "commit =" in line:
            print("Updating GenAI commit in llm_engine.bzl")
            indentation = line[:line.index("commit =")]
            new_line = f'{indentation}commit = "{commit}", # master {date_formatted}\n'
            print(new_line.lstrip())
            new_llm_engine_bzl_content.append(new_line)
        else:
            new_llm_engine_bzl_content.append(line)
    
    with open(llm_engine_bzl_path, 'w') as file:
        file.writelines(new_llm_engine_bzl_content)
    
    # Fetch information about the tokenizers submodule
    tokenizers_commit = None
    tokenizers_commit_date = None
    api_url = f"https://api.github.com/repos/openvinotoolkit/openvino.genai/git/trees/{commit}?recursive=1"
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        submodules = [item for item in data['tree'] if item['type'] == 'commit']
        for submodule in submodules:
            if "openvino_tokenizers" in submodule["path"]:
                tokenizers_commit = submodule["sha"]
                print(f"Tokenizers submodule commit: {tokenizers_commit}")
                
                # Fetch the commit date
                commit_url = f"https://api.github.com/repos/openvinotoolkit/openvino_tokenizers/commits/{tokenizers_commit}"
                commit_response = requests.get(commit_url)
                if commit_response.status_code == 200:
                    commit_data = commit_response.json()
                    commit_date_str = commit_data['commit']['committer']['date']
                    tokenizers_commit_date = commit_date_str.split('T')[0]
                    print(f"Tokenizers submodule commit date: {tokenizers_commit_date}")
                else:
                    print(f"Failed to fetch tokenizers commit date. Status code: {commit_response.status_code}")
    else:
        print(f"Failed to fetch submodules. Status code: {response.status_code}")
    
    # Read the Makefile
    makefile_path = 'Makefile'
    with open(makefile_path, 'r') as file:
        makefile_content = file.readlines()

    # Update Makefile content with OpenVINO Tokenizers commit
    new_makefile_content = []
    for line in makefile_content:
        if "OV_TOKENIZERS_BRANCH ?=" in line:
            print("Updating OV_TOKENIZERS_BRANCH")
            indentation = line[:line.index("OV_TOKENIZERS_BRANCH ?=")]
            new_line = f'{indentation}OV_TOKENIZERS_BRANCH ?= {tokenizers_commit} # master {tokenizers_commit_date}\n'
            print(new_line.lstrip())
            new_makefile_content.append(new_line)
        else:
            new_makefile_content.append(line)

    with open(makefile_path, 'w') as file:
        file.writelines(new_makefile_content)

    model_export_requirements_path = "demos/common/export_models/requirements.txt"
    # Extract the package version from the package link in format <release_version>.dev<date>
    version_match = re.search(r'openvino_genai_[^_]+_(\d+\.\d+\.\d+\.\d+)', windows_package_link)
    if version_match:
        package_version = version_match.group(1)
        print(f"Extracted package version: {package_version}")
    else:
        raise Exception("No package version found in the package link.")

    # Extract commit date from the tokenizers commit
    if tokenizers_commit_date:
        package_version_with_date = f"{package_version}.dev{tokenizers_commit_date.replace('-', '')}"
        print(f"Expanded package version: {package_version_with_date}")
    else:
        raise Exception("Tokenizers commit date not found")
    
    with open(model_export_requirements_path, 'r') as file:
        model_export_requirements_content = file.readlines()

    # Update export model script requirements file with OpenVINO Tokenizers package version
    new_model_export_requirements_content = []
    for line in model_export_requirements_content:
        if "openvino-tokenizers[transformers]<=" in line:
            print("Updating openvino-tokenizers[transformers]<= in requirements.txt")
            new_line = f"openvino-tokenizers[transformers]<={package_version_with_date}\n"
            print(new_line)
            new_model_export_requirements_content.append(new_line)
        else:
            new_model_export_requirements_content.append(line)
    
    with open(model_export_requirements_path, 'w') as file:
        file.writelines(new_model_export_requirements_content)

    # Find a link ending with ".zip" in package_links
    zip_link = next((link for link in package_links if link.endswith('.zip')), None)

    if zip_link:
        # Break the link into two variables
        before_last_slash, after_last_slash = zip_link.rsplit('/', 1)
    else:
        raise Exception("No .zip link found in package_links.")

    # Update windows_install_build_dependencies.bat with new GenAI package link
    win_build_bat_path = "windows_install_build_dependencies.bat"
    with open(win_build_bat_path, 'r') as file:
        win_build_bat_content = file.readlines()
        new_win_build_bat_content = []

    for line in win_build_bat_content:
        if 'set "genai_dir=' in line:
            print("Updating genai_dir in windows_install_build_dependencies.bat")
            new_line = f'set "genai_dir={after_last_slash[:-4]}"\n'
            print(new_line)
            new_win_build_bat_content.append(new_line)
        elif 'set "genai_ver=' in line:
            print("Updating genai_ver in windows_install_build_dependencies.bat")
            new_line = f'set "genai_ver={after_last_slash}"\n'
            print(new_line)
            new_win_build_bat_content.append(new_line)
        elif 'set "genai_http=' in line:
            print("Updating genai_http in windows_install_build_dependencies.bat")
            new_line = f'set "genai_http={before_last_slash}/"\n'
            print(new_line)
            new_win_build_bat_content.append(new_line)
        else:
            new_win_build_bat_content.append(line)
    
    with open(win_build_bat_path, 'w') as file:
        file.writelines(new_win_build_bat_content)


update_openvino()
update_openvino_genai()

