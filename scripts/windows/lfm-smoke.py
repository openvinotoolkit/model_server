"""Owned, loopback-only LFM model/protocol/benchmark evidence on pinned RC4 runtime."""
import argparse
import concurrent.futures
import ctypes
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[2]
MODELS = ('LFM2-24B-A2B-int4-ov', 'LFM2.5-8B-A1B-int4-ov')
LOCK = Path(r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\RC4-LOCAL-INPUTS.lock.json')

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

def powershell(code):
    result = subprocess.run(['powershell', '-NoProfile', '-Command', code], capture_output=True,
                            text=True, creationflags=subprocess.CREATE_NO_WINDOW, check=True)
    return result.stdout.strip()

class MemoryStatus(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in
        ('total_physical', 'available_physical', 'total_pagefile', 'available_pagefile', 'total_virtual', 'available_virtual', 'extended_virtual')]

def free_mib():
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise OSError('GlobalMemoryStatusEx failed')
    return status.available_physical // (1024 * 1024)

class ProcessMemory(ctypes.Structure):
    _fields_ = [('length', ctypes.c_ulong), ('page_faults', ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in
        ('peak_working_set', 'working_set', 'peak_paged_pool', 'paged_pool', 'peak_nonpaged_pool', 'nonpaged_pool',
         'pagefile', 'peak_pagefile', 'private')]

def process_memory(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.CloseHandle.argtypes = (ctypes.c_void_p,)
    handle = kernel.OpenProcess(0x0410, False, pid)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    try:
        query = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
        query.argtypes = (ctypes.c_void_p, ctypes.POINTER(ProcessMemory), ctypes.c_ulong)
        query.restype = ctypes.c_int
        counters = ProcessMemory()
        counters.length = ctypes.sizeof(counters)
        if not query(handle, ctypes.byref(counters), counters.length): raise ctypes.WinError(ctypes.get_last_error())
        return {'PrivateMemorySize64': counters.private, 'WorkingSet64': counters.working_set,
                'PeakWorkingSet64': counters.peak_working_set}
    finally:
        kernel.CloseHandle(handle)

def post(url, body, path):
    encoded = json.dumps(body).encode()
    write(path.with_suffix(path.suffix + '.request.json'), body)
    request = urllib.request.Request(url, data=encoded, headers={'Content-Type': 'application/json'})
    started = time.perf_counter()
    first = last = None
    chunks, events = [], []
    try:
        response = urllib.request.urlopen(request, timeout=300)
    except urllib.error.HTTPError as error:
        path.write_bytes(error.read())
        raise
    with response:
        if body.get('stream'):
            for line in response:
                chunks.append(line)
                if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]': continue
                event = json.loads(line[5:])
                events.append(event)
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    if delta.get('content') or delta.get('reasoning_content') or delta.get('tool_calls'):
                        now = time.perf_counter()
                        if first is None: first = now
                        last = now
            raw = b''.join(chunks)
            assert b'[DONE]' in raw, 'Missing SSE termination'
            usage = next((e['usage'] for e in reversed(events) if e.get('usage')), {})
            payload = events
        else:
            raw = response.read()
            payload = json.loads(raw)
            usage = payload.get('usage', {})
    elapsed = time.perf_counter() - started
    path.write_bytes(raw)
    count = usage.get('completion_tokens')
    metrics = {'wall_seconds': elapsed, 'completion_tokens': count, 'usage': usage,
               'ttft_seconds': first - started if first else None,
               'wall_tps': count / elapsed if count else None,
               'decode_tps_client_observed': (count - 1) / (last - first) if count and count > 1 and first and last > first else None,
               'method': 'client SSE first/last nonempty delta; completion_tokens from server usage; not internal token timestamps'}
    write(path.with_suffix(path.suffix + '.metrics.json'), metrics)
    return payload, metrics

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--device', choices=('CPU', 'GPU'), required=True)
    ap.add_argument('--binary', type=Path, required=True)
    ap.add_argument('--runtime-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--template', type=Path)
    ap.add_argument('--port', type=int, default=18147)
    ap.add_argument('--benchmark', action='store_true')
    ap.add_argument('--plan-only', action='store_true')
    args = ap.parse_args()
    model = Path(r'C:\llm\models') / args.model
    state = {'model': 'OpenVINO/' + args.model, 'local_dir': str(model), 'device': args.device,
             'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'status': 'RUNNING',
             'offload': 'DISABLED; OFFLOAD_RATIO=0', 'gates': {}, 'benchmark': [],
             'source_head': subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()}
    args.output.mkdir(parents=True, exist_ok=False)
    process, watcher = None, None
    gpu_slot = Path(r'C:\gm\qwenmonster\benchmarks\GPU-RUNTIME.lock')
    owns_gpu_slot = False
    stop = threading.Event()
    memory_samples = []
    try:
        lock = json.loads(LOCK.read_text(encoding='utf-8-sig'))
        allowed_dll_hashes = {row['sha256'] for row in lock['variants']['off']['inputs'] if Path(row['source']).suffix.lower() == '.dll'}
        allowed_dll_hashes.add('8df6d149c80841d6eaeb78acb526530823992f1fd7916451da40f480b9cb5162')
        runtime = []
        for row in lock['variants']['off']['inputs']:
            name = Path(row['source']).name
            if name not in ('openvino.dll', 'openvino_genai.dll', 'openvino_tokenizers.dll', 'openvino_intel_gpu_plugin.dll'): continue
            path = args.runtime_dir / name
            digest = sha(path)
            assert digest == row['sha256'], 'Runtime drift: ' + str(path)
            runtime.append({'path': str(path), 'sha256': digest})
        state['runtime'] = runtime
        state['binary'] = {'path': str(args.binary), 'sha256': sha(args.binary)}
        state['model_weight_sha256'] = sha(model / 'openvino_model.bin')
        state['initial_free_mib'] = free_mib()
        # Admission is based on actual bytes and available system RAM, never GPU budget + RAM.
        admission = (model / 'openvino_model.bin').stat().st_size // (1024 * 1024) + 2560
        state['admission_floor_mib'] = admission
        if state['initial_free_mib'] < admission:
            state['status'] = 'BLOCKED'
            state['blocker'] = 'Host RAM admission floor'
            state['gates']['model_load'] = 'NOT_RUN'
            return 2
        if int(powershell('@(Get-Process ovms -ErrorAction SilentlyContinue).Count')):
            state['status'] = 'BLOCKED'
            state['blocker'] = 'Existing OVMS process; no interference allowed'
            return 2
        with socket.socket() as probe:
            probe.bind(('127.0.0.1', args.port))
        serving = model
        if args.template:
            # Only model BINs are hard-linked, always opened read-only by this runner.
            # Config/XML/template files are copies, so an override cannot edit the download.
            serving = args.output / 'serving-model'
            serving.mkdir()
            for source in model.iterdir():
                if not source.is_file(): continue
                target = serving / source.name
                if source.suffix == '.bin': os.link(source, target)
                else: target.write_bytes(source.read_bytes())
            (serving / 'chat_template.jinja').write_bytes(args.template.read_bytes())
            state['template_override'] = {'path': str(args.template), 'sha256': sha(args.template)}
        command = [str(args.binary), '--rest_port', str(args.port), '--rest_bind_address', '127.0.0.1',
                   '--model_path', str(serving), '--model_name', 'lfm', '--task', 'text_generation',
                   '--pipeline_type', 'VLM', '--target_device', args.device, '--tool_parser', 'lfm2',
                   '--enable_tool_guided_generation', 'true', '--max_num_seqs', '4',
                   '--max_num_batched_tokens', '1024', '--cache_size', '0.25',
                   '--enable_prefix_caching', 'false', '--log_level', 'INFO']
        if args.device == 'GPU': command += ['--plugin_config', '{"OFFLOAD_RATIO":0}']
        if args.model.startswith('LFM2.5'): command += ['--reasoning_parser', 'lfm2']
        state['command'] = command
        write(args.output / 'command.json', command)
        if args.plan_only:
            state['status'] = 'NOT_RUN'
            return 0
        if args.device == 'GPU':
            gpu_slot.parent.mkdir(parents=True, exist_ok=True)
            try:
                fd = os.open(gpu_slot, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                state['status'] = 'BLOCKED'
                state['blocker'] = 'GPU runtime slot already reserved: ' + str(gpu_slot)
                return 2
            with os.fdopen(fd, 'w') as stream:
                json.dump({'pid': os.getpid(), 'task': 'lfm-family-hardening', 'output': str(args.output)}, stream)
            owns_gpu_slot = True
        state['hardware'] = json.loads(powershell('Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,PNPDeviceID | ConvertTo-Json'))
        env = {k: v for k, v in os.environ.items() if k not in ('PYTHONHOME', 'PYTHONPATH', 'OPENVINO_LIB_PATHS', 'OpenVINO_DIR')}
        env['PATH'] = str(args.runtime_dir) + r';C:\Windows\System32;C:\Windows'
        with (args.output / 'server.log').open('wb') as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=args.output,
                                       env=env, creationflags=subprocess.CREATE_NO_WINDOW)
            state['owned_pid'] = process.pid
            def guard():
                while not stop.wait(1):
                    available = free_mib()
                    sample = {'monotonic_seconds': time.perf_counter(), 'free_mib': available}
                    try: sample.update(process_memory(process.pid))
                    except OSError: pass  # Process may have exited between the host and process queries.
                    memory_samples.append(sample)
                    if available < 1536:
                        state['memory_guard_triggered'] = True
                        process.terminate()
                        return
            watcher = threading.Thread(target=guard, daemon=True)
            watcher.start()
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if process.poll() is not None: raise RuntimeError('Owned server exited: ' + str(process.returncode))
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/v2/health/ready', timeout=2) as response:
                        if response.status == 200: break
                except (urllib.error.URLError, TimeoutError): pass
                stop.wait(1)
            else: raise TimeoutError('Model readiness timeout')
            state['gates']['model_load'] = 'PASS'
            modules = json.loads(powershell(f'@((Get-Process -Id {process.pid}).Modules | Select-Object ModuleName,FileName) | ConvertTo-Json'))
            if isinstance(modules, dict): modules = [modules]
            loaded = []
            for row in modules:
                name = row['ModuleName'].lower()
                if not name.startswith(('openvino', 'tbb', 'ovms')): continue
                path = Path(row['FileName'])
                loaded.append({**row, 'sha256': sha(path)})
                if name.startswith(('openvino', 'tbb')):
                    assert loaded[-1]['sha256'] in allowed_dll_hashes, 'Unpinned loaded DLL: ' + str(path)
                if name in ('openvino.dll', 'openvino_genai.dll', 'openvino_tokenizers.dll', 'openvino_intel_gpu_plugin.dll'):
                    assert any(r['sha256'] == loaded[-1]['sha256'] for r in runtime), 'Unpinned loaded DLL: ' + str(path)
            write(args.output / 'loaded-modules.json', loaded)
            state['gates']['loaded_modules'] = 'PASS'
            url = f'http://127.0.0.1:{args.port}/v3/chat/completions'
            basic = {'model': 'lfm', 'messages': [{'role': 'user', 'content': 'Reply with the word READY.'}], 'temperature': 0, 'seed': 1234, 'max_tokens': 128}
            response, _ = post(url, basic, args.output / 'unary.json')
            message = response['choices'][0]['message']
            assert message.get('content'), 'No unary content'
            assert '<think>' not in message.get('content', ''), 'Reasoning marker leaked to content'
            state['observed_unary_reasoning'] = bool(message.get('reasoning_content'))
            state['gates']['unary'] = 'PASS'
            streaming = {**basic, 'stream': True, 'stream_options': {'include_usage': True}}
            events, _ = post(url, streaming, args.output / 'sse.txt')
            assert any(c.get('delta', {}).get('content') for e in events for c in e.get('choices', [])), 'No SSE content'
            state['gates']['sse'] = 'PASS'
            tools = [{'type': 'function', 'function': {'name': 'report_value', 'description': 'Report a supplied string exactly.',
                     'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}}, 'required': ['text']}}}]
            target = "O'Brien at C:\\test"
            history = [{'role': 'user', 'content': 'Call report_value once with this exact text: ' + target}]
            tool_request = {**basic, 'messages': history, 'tools': tools,
                            'tool_choice': {'type': 'function', 'function': {'name': 'report_value'}}, 'max_tokens': 256}
            response, _ = post(url, tool_request, args.output / 'tool-unary.json')
            assistant = response['choices'][0]['message']
            calls = assistant.get('tool_calls', [])
            assert len(calls) == 1 and calls[0]['function']['name'] == 'report_value', 'Tool call missing'
            assert json.loads(calls[0]['function']['arguments']) == {'text': target}, 'Tool string changed'
            state['gates']['tool_call'] = 'PASS'
            history += [assistant, {'role': 'tool', 'tool_call_id': calls[0]['id'], 'content': 'TOOL_OK. Tell the user TOOL_OK.'}]
            response, _ = post(url, {**basic, 'messages': history, 'tools': tools, 'tool_choice': 'none', 'max_tokens': 256}, args.output / 'continuation.json')
            assert 'TOOL_OK' in response['choices'][0]['message'].get('content', ''), 'Continuation missing tool result'
            state['gates']['tool_cycle_continuation'] = 'PASS'
            events, _ = post(url, {**tool_request, 'stream': True, 'stream_options': {'include_usage': True}}, args.output / 'tool-sse.txt')
            streamed_calls = {}
            for event in events:
                for choice in event.get('choices', []):
                    for delta in choice.get('delta', {}).get('tool_calls', []):
                        call = streamed_calls.setdefault(delta['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                        call['id'] += delta.get('id', '')
                        call['function']['name'] += delta.get('function', {}).get('name', '')
                        call['function']['arguments'] += delta.get('function', {}).get('arguments', '')
            assert len(streamed_calls) == 1, 'SSE tool call missing'
            streamed_call = next(iter(streamed_calls.values()))
            assert streamed_call['function']['name'] == 'report_value'
            assert json.loads(streamed_call['function']['arguments']) == {'text': target}, 'SSE tool string changed'
            state['gates']['tool_sse'] = 'PASS'
            stream_history = [tool_request['messages'][0], {'role': 'assistant', 'content': '', 'tool_calls': [streamed_call]},
                              {'role': 'tool', 'tool_call_id': streamed_call['id'], 'content': 'TOOL_OK. Tell the user TOOL_OK.'}]
            events, _ = post(url, {**streaming, 'messages': stream_history, 'tools': tools, 'tool_choice': 'none', 'max_tokens': 256}, args.output / 'continuation-sse.txt')
            continued = ''.join(c.get('delta', {}).get('content', '') for e in events for c in e.get('choices', []))
            assert 'TOOL_OK' in continued, 'SSE continuation missing tool result'
            state['gates']['tool_cycle_continuation_sse'] = 'PASS'
            if args.benchmark:
                # The preceding protocol calls are labeled warmup, never included below.
                for length in (128, 512):
                    for size, prompt in [('short', 'Write a detailed numbered description of a forest.'), ('long', ('A forest contains trees, animals and streams. ' * 300) + 'Describe this forest in detail.')]:
                        for concurrency in (1, 2, 4):
                            if free_mib() < 2048: continue
                            for repetition in range(3):
                                body = {**streaming, 'messages': [{'role': 'user', 'content': prompt}], 'max_tokens': length}
                                def one(index):
                                    path = args.output / f'bench-{size}-{length}-c{concurrency}-r{repetition}-i{index}.txt'
                                    _, metrics = post(url, body, path)
                                    return metrics
                                batch_start = time.perf_counter()
                                with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                                    metrics = list(executor.map(one, range(concurrency)))
                                state['benchmark'].append({'input': size, 'max_tokens': length, 'concurrency': concurrency,
                                                           'repetition': repetition, 'batch_wall_seconds': time.perf_counter() - batch_start,
                                                           'free_mib': free_mib(), 'process_memory': process_memory(process.pid), 'requests': metrics})
                complete = bool(state['benchmark']) and all(
                    r['completion_tokens'] and r['ttft_seconds'] and r['decode_tps_client_observed']
                    for batch in state['benchmark'] for r in batch['requests'])
                state['gates']['benchmark'] = 'PASS' if complete else 'FAIL'
                assert complete, 'Missing server token usage or client timing; no verified throughput'
            state['process_memory'] = process_memory(process.pid)
            state['gpu_memory'] = 'NOT_MEASURED; no unverified DXGI/COM probe used'
            state['status'] = 'PASS'
    except Exception as error:
        state['status'] = 'FAIL'
        state['error'] = repr(error)
    finally:
        stop.set()
        if process and process.poll() is None and 'process_memory' not in state:
            try:
                state['process_memory'] = process_memory(process.pid)
            except Exception as memory_error:
                state['process_memory_error'] = repr(memory_error)
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if watcher: watcher.join(timeout=2)
        if owns_gpu_slot:
            saved = json.loads(gpu_slot.read_text())
            if saved.get('pid') == os.getpid(): gpu_slot.unlink()
        state['free_mib_samples'] = memory_samples
        state['owned_process_stopped'] = process is None or process.poll() is not None
        write(args.output / 'RESULT.json', state)
        print(json.dumps({'output': str(args.output), 'status': state['status'], 'gates': state['gates'],
                          'error': state.get('error'), 'blocker': state.get('blocker')}, indent=2))
    return 1 if state['status'] == 'FAIL' else 0

if __name__ == '__main__':
    sys.exit(main())
