"""Pinned RC4 adaptation of windows_create_package.bat; never alters installed OVMS."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile

REPO = Path(__file__).resolve().parents[2]
LOCK = REPO / 'docs/qwenmonster/testing/fixed-package-20261024/speed-comparison-20261024/CANDIDATE-CUSTOM-GPU.lock.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def copy(source, dest, expected=None):
    if expected and (source.stat().st_size != expected['bytes'] or sha(source) != expected['sha256']):
        raise ValueError('Pinned input changed: ' + str(source))
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and sha(source) != sha(dest):
        raise ValueError('Conflicting package files: ' + str(dest))
    shutil.copy2(source, dest)


def run(command, env, cwd, log):
    result = subprocess.run(list(map(str, command)), env=env, cwd=cwd, capture_output=True, text=True,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    log.write_text(result.stdout + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{command[0]} failed ({result.returncode}): {log}')
    return result.stdout


def python_backend_smoke(package, env, root):
    """Exercise the compiled PythonExecutorCalculator, not only an importable binding."""
    folder = root / 'python-backend-smoke'
    folder.mkdir(exist_ok=False)
    handler = folder / 'handler.py'
    handler.write_text("from pyovms import Tensor\nclass OvmsPythonModel:\n"
                       "    def initialize(self, kwargs): pass\n"
                       "    def execute(self, inputs, kwargs={}):\n"
                       "        return [Tensor('out', inputs[0].data.tobytes())]\n", encoding='utf-8')
    graph = folder / 'graph.pbtxt'
    graph.write_text('input_stream: "OVMS_PY_TENSOR:in"\noutput_stream: "OVMS_PY_TENSOR:out"\n'
        'node { name: "python_probe" calculator: "PythonExecutorCalculator"\n'
        'input_side_packet: "PYTHON_NODE_RESOURCES:py" input_stream: "INPUT:in" output_stream: "OUTPUT:out"\n'
        'node_options { [type.googleapis.com/mediapipe.PythonExecutorCalculatorOptions] { handler_path: '
        + json.dumps(handler.as_posix()) + ' } } }\n', encoding='utf-8')
    config = folder / 'config.json'
    write(config, {'model_config_list': [], 'mediapipe_config_list': [{'name': 'rc4_python', 'graph_path': graph.as_posix()}]})
    with (folder / 'server.log').open('wb') as output:
        process = subprocess.Popen([str(package / 'ovms.exe'), '--config_path', str(config),
                  '--rest_port', '18138', '--rest_bind_address', '127.0.0.1', '--file_system_poll_wait_seconds', '0'],
                  env=env, cwd=folder, stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            deadline = time.monotonic() + 45
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Python backend server exited; inspect ' + str(folder))
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18138/v2/models/rc4_python/ready', timeout=2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError('Python backend readiness timeout')
                    time.sleep(0.5)
            body = {'inputs': [{'name': 'in', 'shape': [3], 'datatype': 'UINT8', 'data': [114, 99, 52]}]}
            write(folder / 'request.json', body)
            request = urllib.request.Request('http://127.0.0.1:18138/v2/models/rc4_python/infer',
                      json.dumps(body).encode(), {'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=10) as response:
                result = json.load(response)
            write(folder / 'response.json', result)
            assert result['outputs'][0]['name'] == 'out' and result['outputs'][0]['data'] == [114, 99, 52], result
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
    write(folder / 'RESULT.json', {'status': 'PASS', 'check': 'PythonExecutorCalculator UINT8 HTTP round trip'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', choices=['on', 'off'], required=True)
    parser.add_argument('--root', type=Path, default=Path(r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4'))
    args = parser.parse_args()
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    root = args.root.resolve()
    build = root / ('build-py_' + args.python)
    receipt = read(build / 'COMPLETE.json')
    for row in receipt['files']:
        p = build / row['path']
        if sha(p) != row['sha256']:
            raise ValueError('Build snapshot changed: ' + str(p))
    stage = root / ('stage-py_' + args.python) / 'ovms'
    stage.mkdir(parents=True, exist_ok=False)
    lock = read(LOCK)
    # Same flat ovms/ layout as the upstream Windows package, with explicit pinned input paths.
    for row in lock['payload_files']:
        relative = Path(row['destination'])
        if relative.parts[0] == 'bin':
            source = Path(row['source'])
            if source.name.lower() == 'ovms.exe':
                source = build / 'bin/ovms.exe'
                copy(source, stage / source.name)
            elif source.parent.name == 'src':
                # Retain the selected measured helper cohort even if CMake regenerates a DLL.
                copy(root / 'measured-original/bin' / source.name, stage / source.name, row)
            else:
                copy(source, stage / source.name, row)
        elif relative.parts[0] == 'licenses':
            copy(Path(row['source']), stage / 'thirdparty-licenses' / Path(*relative.parts[1:]), row)
        else:
            copy(Path(row['source']), stage / relative, row)
    for row in lock['payload_trees']:
        if row['destination'] == 'python':
            continue  # Build-helper Python is not the release runtime.
        source = Path(row['source'])
        spec = importlib.util.spec_from_file_location('input_verifier', REPO / 'docs/qwenmonster/testing/fixed-package-20261024/verify-inputs.py')
        verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(verifier)
        if verifier.tree_pin(source) != (row['files'], row['bytes'], row['tree_sha256']):
            raise ValueError('License input tree changed: ' + str(source))
        # Keep license namespaces to prevent collisions while including every upstream license.
        destination = stage / 'thirdparty-licenses' / Path(row['destination']).name
        shutil.copytree(source, destination)
    for name in ['setupvars.bat', 'setupvars.ps1', 'install_ovms_service.bat']:
        copy(REPO / name, stage / name)
    (stage / 'RELEASE-NOTES.txt').write_text(
        'GEMMAMONSTER 2026.4 RC4, Windows x86_64, Python ' + args.python + '\n'
        'Build source: ' + receipt['source_commit'] + '\n'
        'Run setupvars.ps1 (PowerShell) or setupvars.bat (cmd), then ovms.exe --help.\n'
        'Models are not bundled. Requires a compatible Intel graphics driver and VC++ x64 runtime.\n'
        'Unsigned community fork; custom GPU source-to-binary attestation not reestablished.\n'
        'MANIFEST.json records every payload file. See the RC4 release for validation and limits.\n'
        'https://github.com/DassaultFalconKing/gemmamonster_model_server_OVMS/releases/tag/gemmamonster-2026.4-rc4\n',
        encoding='utf-8')
    downloads = root / 'downloads'
    downloads.mkdir(exist_ok=True)
    if args.python == 'on':
        download_lock = read(REPO / 'docs/qwenmonster/testing/rc4-release/PYTHON-DOWNLOADS.lock.json')
        if {p.name for p in downloads.iterdir() if p.is_file()} != {row['name'] for row in download_lock}:
            raise ValueError('Python download directory contains unlocked or missing artifacts')
        for row in download_lock:
            p = downloads / row['name']
            if not p.is_file() or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                raise ValueError('Pinned Python download missing or changed: ' + str(p))
        archive = downloads / 'python-3.12.10-embed-amd64.zip'
        python = stage / 'python'
        python.mkdir()
        with zipfile.ZipFile(archive) as z:
            z.extractall(python)
        # Equivalent to windows_prepare_python.bat: expanded stdlib and explicit isolated paths.
        with zipfile.ZipFile(python / 'python312.zip') as z:
            z.extractall(python / 'python312')
        (python / 'python312.zip').unlink()
        (python / 'python312._pth').write_text('./python312\n.\n./Scripts\n./Lib/site-packages\nimport site\n')
        copy(build / 'pyovms.pyd', python / 'pyovms.pyd')
        run([sys.executable, '-m', 'pip', '--python', python / 'python.exe', 'install', '--no-index', '--no-compile',
             '--find-links', downloads, 'pip==25.0.1', 'setuptools==80.9.0', 'Jinja2==3.1.6', 'MarkupSafe==3.0.2'],
            os.environ.copy(), root, root / 'python-install.log')
    dumpbin = Path(r'C:\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\dumpbin.exe')
    import_scan = []
    unresolved = []
    package_names = {p.name.lower() for p in stage.rglob('*') if p.is_file()}
    system32 = Path(os.environ['SystemRoot']) / 'System32'
    for p in sorted(stage.rglob('*')):
        if p.suffix.lower() not in ('.exe', '.dll', '.pyd'):
            continue
        result = subprocess.run([str(dumpbin), '/DEPENDENTS', str(p)], capture_output=True, text=True,
                                creationflags=subprocess.CREATE_NO_WINDOW, check=True)
        deps = []
        for name in sorted(set(re.findall(r'^\s+([\w.\-]+\.dll)\s*$', result.stdout, re.M | re.I))):
            if name.lower() in package_names:
                resolution = 'package'
            elif name.lower().startswith(('api-ms-', 'ext-ms-')) or (system32 / name).exists():
                resolution = 'Windows host prerequisite'
            else:
                resolution = 'UNRESOLVED'
                unresolved.append(p.name + ': ' + name)
            deps.append({'name': name, 'resolution': resolution})
        import_scan.append({'file': p.relative_to(stage).as_posix(), 'dependencies': deps})
    write(root / ('STATIC-py_' + args.python + '.json'), {'status': 'FAIL' if unresolved else 'PASS',
          'files': import_scan, 'unresolved': unresolved, 'scope': 'Static PE imports; dynamic loads checked separately'})
    if unresolved:
        raise ValueError('Unresolved static imports: ' + repr(unresolved))
    # Public manifest has hashes and source identities, never host environment variables.
    manifest = {'release': 'gemmamonster-2026.4-rc4', 'python': args.python,
                'source_commit': receipt['source_commit'], 'config': receipt['config'],
                'sdk_archive_sha256': lock['sdk_archive']['sha256'],
                'helper_policy': 'Frozen measured git2/curl/OpenCV cohort; copied from preserved original snapshot and hash-checked',
                'custom_gpu_source_attestation': 'NOT_REESTABLISHED; retained measured binary 21a2fd9f',
                'signing': 'Unsigned community fork package; upstream DLL signatures retained',
                'files': []}
    for p in sorted(stage.rglob('*')):
        if p.is_file():
            manifest['files'].append({'path': p.relative_to(stage).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    write(stage / 'MANIFEST.json', manifest)
    assets = root / 'assets'
    assets.mkdir(exist_ok=True)
    name = 'gemmamonster-2026.4-rc4-windows-x86_64-python_' + args.python
    write(assets / (name + '.manifest.json'), manifest)
    archive = assets / (name + '.zip')
    if archive.exists():
        raise FileExistsError(archive)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(stage.rglob('*')):
            if p.is_file():
                z.write(p, 'ovms/' + p.relative_to(stage).as_posix())
    extracted = root / ('extracted-py_' + args.python)
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise ValueError('Archive CRC failed')
        z.extractall(extracted)
    package = extracted / 'ovms'
    for row in manifest['files']:
        if sha(package / row['path']) != row['sha256']:
            raise ValueError('Extracted hash mismatch: ' + row['path'])
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'OpenVINO_DIR', 'OVMS_DIR'):
        env.pop(key, None)
    env['PATH'] = str(package) + ';' + os.environ['SystemRoot'] + r'\System32;' + os.environ['SystemRoot']
    if args.python == 'on':
        env['PYTHONHOME'] = str(package / 'python')
        env['PATH'] = str(package / 'python') + ';' + env['PATH']
    version = run([package / 'ovms.exe', '--version'], env, package, root / ('version-py_' + args.python + '.log'))
    if '2026.4.0-rc4' not in version or receipt['config'] not in version:
        raise ValueError('Unexpected binary version: ' + version)
    run([package / 'ovms.exe', '--help'], env, package, root / ('help-py_' + args.python + '.log'))
    # ovms_test has a Python-initializing global test environment even for py_off.
    # This build/test dependency is intentionally separate from the shipped server environment.
    test_env = env.copy()
    test_python = Path(r'C:\gm\qwenmonster\semantic-refit-2026.4-tools\py-nupkg\tools')
    test_env['PYTHONHOME'] = str(test_python)
    test_env['PATH'] = str(test_python) + ';' + env['PATH']
    if args.python == 'on':
        test_env['PYTHONPATH'] = str(package / 'python/Lib/site-packages') + ';' + str(package / 'python')
    run([build / 'bin/ovms_test.exe', '--gtest_filter=*Qwen3Coder*',
         '--gtest_output=xml:' + str(root / ('qwen-py_' + args.python + '.xml'))], test_env, REPO,
        root / ('qwen-py_' + args.python + '.log'))
    if args.python == 'on':
        code = "import os, sys; assert sys.version_info[:3]==(3,12,10); dll_dir=os.add_dll_directory(r'" + str(package) + "'); import pyovms, jinja2, markupsafe; t=pyovms.Tensor('probe', b'rc4'); assert t.data.tobytes()==b'rc4'; print('PYOVMS_PASS', sys.version.split()[0], t.name, t.shape)"
        run([package / 'python/python.exe', '-B', '-c', code], env, package, root / 'python-binding-smoke.log')
        python_backend_smoke(package, env, root)
    write(root / ('PACKAGE-py_' + args.python + '.json'), {'status': 'PASS', 'archive': archive.name,
          'sha256': sha(archive), 'bytes': archive.stat().st_size, 'files_verified': len(manifest['files']),
          'checks': ['ZIP CRC', 'extracted hashes', 'isolated PATH version/help', '*Qwen3Coder*'] +
          (['embedded Python pyovms Tensor', 'PythonExecutorCalculator HTTP round trip'] if args.python == 'on' else []),
          'version': version, 'not_run': ['full OVMS suite', 'long context', 'soak', 'agent loops']})
    print(archive)


if __name__ == '__main__':
    main()
