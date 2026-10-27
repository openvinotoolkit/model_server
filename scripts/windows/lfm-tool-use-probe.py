"""Raw unary/SSE policy probes; no generated command is executed."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time
import urllib.error
import urllib.request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default='http://127.0.0.1:8137/v3/chat/completions')
    ap.add_argument('--model', default='lfm25-8b-a1b-int4')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--timeout', type=int, default=120)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tools = [{'type': 'function', 'function': {'name': 'cwd', 'description': 'Return the actual current working directory.',
        'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}]
    cases = [
        ('auto-action', 'Call cwd to find the actual current directory. Use the tool; do not give instructions.', 'auto', False),
        ('auto-no-action', 'What is 2 + 2? Answer directly.', 'auto', False),
        ('required-conflict', 'Say hello without using tools.', 'required', False),
        ('named-conflict', 'Say hello without using tools.', {'type': 'function', 'function': {'name': 'cwd'}}, False),
        ('none', 'Call cwd.', 'none', False),
        ('required-sse', 'Call cwd.', 'required', True),
        ('required-no-tools', 'Say hello.', 'required', False),
        ('named-unregistered', 'Say hello.', {'type': 'function', 'function': {'name': 'missing'}}, False),
    ]
    results = []
    for name, prompt, choice, stream in cases:
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': prompt}], 'tools': tools,
                'tool_choice': choice, 'parallel_tool_calls': False, 'temperature': 0, 'seed': 1234,
                'max_tokens': 256, 'stream': stream}
        if name == 'required-no-tools':
            del body['tools']
        encoded = json.dumps(body).encode('utf-8')
        (args.output / (name + '.request.json')).write_bytes(encoded)
        started = time.monotonic()
        status, raw, error = None, b'', None
        try:
            req = urllib.request.Request(args.url, data=encoded, headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(req, timeout=args.timeout) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as exc:
            status, raw = exc.code, exc.read()
        except Exception as exc:
            error = repr(exc)
        (args.output / (name + '.response.txt')).write_bytes(raw)
        calls, content, finishes = {}, '', []
        try:
            events = [json.loads(line[5:]) for line in raw.splitlines() if line.startswith(b'data:') and line[5:].strip() != b'[DONE]'] if stream else [json.loads(raw)]
            for event in events:
                for candidate in event.get('choices', []):
                    msg = candidate.get('delta' if stream else 'message', {})
                    content += msg.get('content') or ''
                    if candidate.get('finish_reason'):
                        finishes.append(candidate['finish_reason'])
                    for call in msg.get('tool_calls', []):
                        index = call.get('index', len(calls) if not stream else 0)
                        target = calls.setdefault(index, {'name': '', 'arguments': '', 'id': ''})
                        for key, val in call.get('function', {}).items():
                            if key in target:
                                target[key] += val
                        if call.get('id'):
                            target['id'] += call['id']
        except Exception as exc:
            error = error or repr(exc)
        valid = bool(calls) and all(c['name'] == 'cwd' and json.loads(c['arguments']) == {} for c in calls.values())
        expected = (valid and len(calls) == 1) if name not in ('auto-no-action', 'none') else not calls
        passed = status == 200 and not error and expected
        if name in ('required-no-tools', 'named-unregistered'):
            passed = status in (400, 422)
        result = {'name': name, 'http': status, 'elapsed_s': time.monotonic() - started,
                  'calls': calls, 'content': content, 'finish_reasons': finishes, 'error': error,
                  'verdict': 'PASS' if passed else 'FAIL',
                  'response_sha256': hashlib.sha256(raw).hexdigest()}
        results.append(result)
        summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'url': args.url,
                   'model': args.model, 'results': results, 'scope': 'API/model protocol only; tool execution NOT_RUN'}
        (args.output / 'RESULT.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(name, result['verdict'], status, 'calls', len(calls), flush=True)


if __name__ == '__main__':
    main()
