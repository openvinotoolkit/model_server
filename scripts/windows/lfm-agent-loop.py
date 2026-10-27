"""Bounded real host-tool loop. Generated shell commands are never executed."""
import argparse
import json
from pathlib import Path
import urllib.request
import urllib.error


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', required=True)
    ap.add_argument('--model', default='lfm25-8b-a1b-int4')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--max-tokens', type=int, default=2048)
    ap.add_argument('--profile', choices=['deterministic', 'official', 'native'], default='deterministic')
    ap.add_argument('--tool-policy', choices=['auto', 'action-required'], default='auto')
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tools = []
    for name, description, props, required in [
        ('cwd', 'Get the actual working directory.', {}, []),
        ('write', 'Create a UTF-8 text file in the working directory.', {'path': {'type': 'string'}, 'content': {'type': 'string'}}, ['path', 'content']),
        ('read', 'Read a UTF-8 text file in the working directory.', {'path': {'type': 'string'}}, ['path']),
    ]:
        tools.append({'type': 'function', 'function': {'name': name, 'description': description,
            'parameters': {'type': 'object', 'properties': props, 'required': required, 'additionalProperties': False}}})
    results = []
    for stream in [False, True]:
        for repeat in range(args.repeats):
            root = args.output / ('sse' if stream else 'unary') / str(repeat)
            sandbox = root / 'sandbox'
            sandbox.mkdir(parents=True, exist_ok=True)
            # Avoid JSON/Python single-backslash traps (\t/\n/\b) that silently alter Windows paths.
            # Same apostrophe + path coverage as lfm-smoke tool_call gate.
            payload = "O'Brien C:/llm/lfm-test 1.10"
            # Agent contract: structured tool_calls only. Prose that names tools does not execute them.
            # parallel_tool_calls=false is intentional — one call per turn, then consume its result.
            messages = [{'role': 'system', 'content': (
                'You are an agent with executable tools. Perform requested file actions using tools, '
                'inspect their results, then report. Never claim an action occurred before its result. '
                'Emit exactly one tool call per assistant turn, then wait for that tool result. '
                'A text description of an action does not execute it. Do not invent tool output.'
            )},
                {'role': 'user', 'content': 'Узнай текущий каталог инструментом cwd, создай hello.txt с точным содержимым ' + json.dumps(payload, ensure_ascii=False) + ', затем прочитай файл инструментом read и сообщи результат. Выполни действия сам.'}]
            if args.profile == 'native':
                messages[0]['content'] += (
                    ' Invoke tools with native calls, for example '
                    '<|tool_call_start|>[cwd()]<|tool_call_end|>. '
                    'Do not put multiple calls inside one envelope.'
                )
            executed = []
            error = None
            for turn in range(8):
                body = {'model': args.model, 'messages': messages, 'tools': tools, 'tool_choice': 'auto',
                    'parallel_tool_calls': False, 'temperature': 0, 'seed': 1234 + repeat, 'max_tokens': args.max_tokens, 'stream': stream}
                if args.profile == 'official':
                    body.update(temperature=0.2, top_k=80, repetition_penalty=1.05)
                if args.profile == 'native':
                    body.update(temperature=0.2, top_k=80, repetition_penalty=1.05)
                # Hard required/named modes are explicit policy — not a forced-auto heuristic.
                if args.tool_policy == 'action-required':
                    body['tool_choice'] = 'none' if [x['tool'] for x in executed] == ['cwd', 'write', 'read'] else 'required'
                (root / f'{turn}.request.json').write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
                try:
                    req = urllib.request.Request(args.url, data=json.dumps(body).encode(), headers={'Content-Type': 'application/json'})
                    with urllib.request.urlopen(req, timeout=180) as response:
                        raw = response.read()
                    (root / f'{turn}.response.txt').write_bytes(raw)
                    if not stream:
                        choice = json.loads(raw)['choices'][0]
                        assistant = choice['message']
                    else:
                        assistant = {'role': 'assistant', 'content': '', 'reasoning_content': '', 'tool_calls': []}
                        indexed = {}
                        for line in raw.splitlines():
                            if not line.startswith(b'data:') or line[5:].strip() == b'[DONE]': continue
                            for choice in json.loads(line[5:]).get('choices', []):
                                delta = choice.get('delta', {})
                                for key in ['content', 'reasoning_content']:
                                    assistant[key] += delta.get(key) or ''
                                for call in delta.get('tool_calls', []):
                                    saved = indexed.setdefault(call['index'], {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                                    saved['id'] += call.get('id') or ''
                                    for key in ['name', 'arguments']:
                                        saved['function'][key] += call.get('function', {}).get(key) or ''
                        assistant['tool_calls'] = list(indexed.values())
                    messages.append(assistant)
                    calls = assistant.get('tool_calls') or []
                    if not calls: break
                    if len(calls) != 1: raise ValueError('parallel=false violated')
                    for call in calls:
                        name = call['function']['name']
                        params = json.loads(call['function']['arguments'])
                        if name == 'cwd' and params == {}:
                            value = {'cwd': str(sandbox.resolve())}
                        elif name in ('write', 'read'):
                            if params.get('path') not in ('hello.txt', str(sandbox.resolve() / 'hello.txt')):
                                raise ValueError('out-of-scope file path')
                            target = sandbox / 'hello.txt'
                            if name == 'write':
                                target.write_text(params['content'], encoding='utf-8')
                                value = {'written': 'hello.txt', 'bytes': target.stat().st_size}
                            else:
                                value = {'path': 'hello.txt', 'content': target.read_text(encoding='utf-8')}
                        else: raise ValueError('unregistered tool or wrong arguments')
                        executed.append({'tool': name, 'arguments': params, 'result': value})
                        messages.append({'role': 'tool', 'tool_call_id': call['id'], 'content': json.dumps(value, ensure_ascii=False)})
                except Exception as exc:
                    if isinstance(exc, urllib.error.HTTPError):
                        (root / f'{turn}.error-response.txt').write_bytes(exc.read())
                    error = repr(exc)
                    break
            names = [x['tool'] for x in executed]
            target = sandbox / 'hello.txt'
            passed = not error and names == ['cwd', 'write', 'read'] and target.exists() and target.read_text(encoding='utf-8') == payload and not (messages[-1].get('tool_calls') or []) and messages[-1]['role'] == 'assistant'
            results.append({'stream': stream, 'repeat': repeat, 'profile': args.profile, 'tool_policy': args.tool_policy,
                            'verdict': 'PASS' if passed else 'FAIL', 'executed': executed, 'error': error, 'final': messages[-1]})
            (args.output / 'RESULT.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
            print('stream', stream, 'repeat', repeat, 'PASS' if passed else 'FAIL', names, error, flush=True)


if __name__ == '__main__':
    main()
