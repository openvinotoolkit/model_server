"""Capture public canonical and peer tool protocol sources at resolved commits."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import urllib.request


def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'LFM-MFP1-evidence'})
    return urllib.request.urlopen(req, timeout=30).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sources = []
    for model in ['LFM2.5-8B-A1B', 'LFM2-24B-A2B']:
        revision = json.loads(get('https://huggingface.co/api/models/LiquidAI/' + model))['sha']
        for filename in ['chat_template.jinja', 'README.md', 'tokenizer_config.json']:
            url = f'https://huggingface.co/LiquidAI/{model}/resolve/{revision}/{filename}'
            raw = get(url)
            dest = args.output / (model + '-' + filename)
            dest.write_bytes(raw)
            sources.append({'source': 'LiquidAI/' + model, 'revision': revision, 'url': url,
                'file': str(dest), 'sha256': hashlib.sha256(raw).hexdigest()})
    for repo, filename in [
        ('vllm-project/vllm', 'vllm/tool_parsers/lfm2_tool_parser.py'),
        ('sgl-project/sglang', 'python/sglang/srt/function_call/lfm2_detector.py'),
        ('ggml-org/llama.cpp', 'common/chat.cpp'),
    ]:
        try:
            branch = json.loads(get(f'https://api.github.com/repos/{repo}'))['default_branch']
            revision = json.loads(get(f'https://api.github.com/repos/{repo}/commits/{branch}'))['sha']
            url = f'https://raw.githubusercontent.com/{repo}/{revision}/{filename}'
            raw = get(url)
            dest = args.output / (repo.replace('/', '-') + '-' + Path(filename).name)
            dest.write_bytes(raw)
            sources.append({'source': repo, 'revision': revision, 'url': url,
                'file': str(dest), 'sha256': hashlib.sha256(raw).hexdigest()})
        except Exception as exc:
            sources.append({'source': repo, 'status': 'NOT_CAPTURED', 'error': repr(exc)})
    (args.output / 'SOURCES.lock.json').write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'sources': sources}, indent=2), encoding='utf-8')
    print(json.dumps(sources, indent=2))


if __name__ == '__main__':
    main()
