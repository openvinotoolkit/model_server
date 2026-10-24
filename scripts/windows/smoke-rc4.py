"""One owned loopback Ornith smoke per extracted RC4 package; no deployment changes."""
import argparse
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import urllib.request


class Memory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(n, ctypes.c_ulonglong) for n in
                ('total', 'available', 'total_page', 'available_page', 'total_virtual', 'available_virtual', 'extended')]


def free():
    m = Memory()
    m.length = ctypes.sizeof(m)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise ctypes.WinError()
    return m.available / 1048576


def write(p, value):
    p.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--python', choices=['on', 'off'], required=True)
    ap.add_argument('--root', type=Path, default=Path(r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4'))
    ap.add_argument('--model', type=Path, default=Path(r'C:\llm\models\ornith-35b-ov'))
    a = ap.parse_args()
    root = a.root.resolve()
    package = root / ('extracted-py_' + a.python) / 'ovms'
    folder = root / ('live-py_' + a.python)
    folder.mkdir(exist_ok=False)
    slot = Path(r'C:\gm\qwenmonster\benchmarks\GPU-RUNTIME.lock')
    fd = os.open(slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, json.dumps({'pid': os.getpid(), 'task': 'rc4-extracted-smoke'}).encode())
    os.close(fd)
    process = None
    stop = threading.Event()
    samples = []
    state = {'status': 'RUNNING', 'python': a.python, 'exe_sha256': sha(package / 'ovms.exe'),
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat()}
    try:
        identity = root / 'MODEL-IDENTITY.json'
        model_files = sorted(p for p in a.model.iterdir() if p.is_file() and p.suffix.lower() in ('.bin', '.xml', '.json', '.jinja'))
        if not identity.exists():
            write(identity, {'model': 'Ornith 35B local OpenVINO export', 'files': [
                {'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': sha(p)} for p in model_files]})
        captured = json.loads(identity.read_text())
        assert {p.name for p in model_files} == {x['name'] for x in captured['files']}, 'Model file set changed'
        for row in captured['files']:
            p = a.model / row['name']
            assert p.stat().st_size == row['bytes'] and p.stat().st_mtime_ns == row['mtime_ns'], 'Model input changed between variants'
        active = subprocess.run(['powershell', '-NoProfile', '-Command',
            '@(Get-Process ovms -ErrorAction SilentlyContinue).Count'], capture_output=True, text=True,
            creationflags=subprocess.CREATE_NO_WINDOW, check=True)
        if int(active.stdout.strip()) != 0:
            state['resource_block'] = 'Existing OVMS process'
            raise RuntimeError('Existing OVMS process: no GPU concurrency')
        state['initial_free_mib'] = free()
        if state['initial_free_mib'] < 15463:
            state['resource_block'] = 'RAM admission floor'
            raise RuntimeError('RAM admission floor 15463 MiB not met')
        env = os.environ.copy()
        for k in ('PYTHONHOME', 'PYTHONPATH', 'OPENVINO_LIB_PATHS', 'OpenVINO_DIR'):
            env.pop(k, None)
        env['PATH'] = str(package) + r';C:\Windows\System32;C:\Windows'
        if a.python == 'on':
            env['PYTHONHOME'] = str(package / 'python')
            env['PATH'] = str(package / 'python') + ';' + env['PATH']
        env['PYTHONDONTWRITEBYTECODE'] = '1'
        command = [str(package / 'ovms.exe'), '--rest_port', '18137', '--rest_bind_address', '127.0.0.1',
                   '--model_path', str(a.model), '--model_name', 'ornith', '--task', 'text_generation',
                   '--pipeline_type', 'VLM', '--target_device', 'GPU', '--tool_parser', 'qwen3coder',
                   '--reasoning_parser', 'qwen3', '--enable_tool_guided_generation', 'true',
                   '--max_num_seqs', '1', '--max_num_batched_tokens', '4096', '--cache_size', '2',
                   '--enable_prefix_caching', 'false', '--log_level', 'INFO']
        write(folder / 'command.json', command)
        with (folder / 'server.log').open('wb') as output:
            process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, cwd=folder, env=env,
                                       creationflags=subprocess.CREATE_NO_WINDOW)
            state['owned_pid'] = process.pid
            def guard():
                while not stop.wait(1):
                    samples.append(free())
                    if samples[-1] < 1024:
                        state['memory_guard'] = True
                        process.terminate()
                        return
            watcher = threading.Thread(target=guard, daemon=True)
            watcher.start()
            deadline = time.monotonic() + 360
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Owned server exited: ' + str(process.returncode))
                try:
                    with urllib.request.urlopen('http://127.0.0.1:18137/v1/models', timeout=2) as response:
                        models = json.load(response)
                    if any(m.get('id') == 'ornith' for m in models.get('data', [])):
                        break
                except OSError:
                    pass
                if time.monotonic() > deadline:
                    raise TimeoutError('Model readiness timeout')
                time.sleep(1)
            bodies = [
                {'model': 'ornith', 'messages': [{'role': 'user', 'content': 'Reply with one short sentence about the moon.'}],
                 'temperature': 0, 'max_tokens': 48, 'stream': True, 'stream_options': {'include_usage': True},
                 'chat_template_kwargs': {'enable_thinking': False}},
                {'model': 'ornith', 'messages': [{'role': 'user', 'content': 'Call record with path C:\\test and tag #B4E8.'}],
                 'temperature': 0, 'max_tokens': 128, 'stream': False, 'tool_choice': 'required',
                 'chat_template_kwargs': {'enable_thinking': False}, 'tools': [{'type': 'function', 'function': {
                     'name': 'record', 'parameters': {'type': 'object', 'properties': {
                         'path': {'type': 'string'}, 'tag': {'type': 'string'}}, 'required': ['path', 'tag']}}}]}]
            for i, body in enumerate(bodies):
                write(folder / f'request-{i}.json', body)
                request = urllib.request.Request('http://127.0.0.1:18137/v3/chat/completions',
                         json.dumps(body).encode(), {'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=180) as response:
                    raw = response.read()
                (folder / f'response-{i}.txt').write_bytes(raw)
                if i == 0:
                    events = [json.loads(line[5:]) for line in raw.splitlines()
                              if line.startswith(b'data:') and line[5:].strip() != b'[DONE]']
                    text = ''.join(c.get('delta', {}).get('content', '') for e in events for c in e.get('choices', []))
                    assert text and b'[DONE]' in raw, 'Missing SSE content/termination'
                    state['sse_text'] = text
                else:
                    calls = json.loads(raw)['choices'][0]['message'].get('tool_calls', [])
                    assert len(calls) == 1 and calls[0]['function']['name'] == 'record', 'Missing tool call'
                    arguments = json.loads(calls[0]['function']['arguments'])
                    assert arguments == {'path': 'C:\\test', 'tag': '#B4E8'}, arguments
                    state['tool_arguments'] = calls[0]['function']['arguments']
            result = subprocess.run(['powershell', '-NoProfile', '-Command',
                f'Get-Process -Id {process.pid} | Select-Object -ExpandProperty Modules | Select-Object ModuleName,FileName | ConvertTo-Json -Compress'],
                capture_output=True, text=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            loaded = []
            for row in json.loads(result.stdout):
                if row['ModuleName'].lower().startswith(('ovms', 'openvino', 'tbb', 'python', 'git2', 'libcurl', 'opencv')):
                    p = Path(row['FileName']).resolve()
                    assert p.is_relative_to(package.resolve()), 'External runtime DLL: ' + str(p)
                    loaded.append({'name': row['ModuleName'], 'package_path': p.relative_to(package).as_posix(), 'sha256': sha(p)})
                elif row['ModuleName'].lower().startswith(('igdrcl', 'igc64', 'ze_intel_gpu')) or row['ModuleName'].lower() == 'opencl.dll':
                    loaded.append({'name': row['ModuleName'], 'resolution': 'host GPU driver prerequisite', 'sha256': sha(Path(row['FileName']))})
            gpu = next(x for x in loaded if x['name'].lower() == 'openvino_intel_gpu_plugin.dll')
            assert gpu['sha256'] == '21a2fd9f5d856c297416059325832f0f902c84a4e78b840f859ca508f5aea66d'
            write(folder / 'loaded-modules.json', loaded)
            state['status'] = 'PASS'
    except Exception as e:
        state['status'] = 'NEEDS_REVIEW' if state.get('memory_guard') or state.get('resource_block') else 'FAIL'
        state['error'] = repr(e)
        raise
    finally:
        stop.set()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        state['minimum_free_mib'] = min(samples, default=free())
        state['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
        write(folder / 'RESULT.json', state)
        slot.unlink()


if __name__ == '__main__':
    main()
