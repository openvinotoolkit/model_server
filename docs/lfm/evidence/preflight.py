"""Read-only LFM input inventory; never loads weights or changes model files."""
import collections
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
RELEASE = Path(r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()

result = {'source_head': git('rev-parse', 'HEAD'), 'models': [], 'runtime': []}
for model in ('LFM2-24B-A2B-int4-ov', 'LFM2.5-8B-A1B-int4-ov'):
    root = Path(r'C:\llm\models') / model
    files = []
    for path in sorted(root.iterdir()):
        if not path.is_file():
            continue
        metadata = root / '.cache/huggingface/download' / (path.name + '.metadata')
        receipt = metadata.read_text().splitlines() if metadata.exists() else []
        digest = sha(path)
        files.append({'name': path.name, 'bytes': path.stat().st_size, 'sha256': digest,
                      'download_revision': receipt[0] if receipt else None,
                      'download_etag': receipt[1] if len(receipt) > 1 else None,
                      'lfs_sha256_matches': digest == receipt[1] if len(receipt) > 1 and len(receipt[1]) == 64 else None})
    graph = ET.parse(root / 'openvino_model.xml').getroot()
    layers = list(graph.find('layers'))
    ops = collections.Counter(layer.get('type') for layer in layers)
    constants = []
    for layer in layers:
        if layer.get('type') != 'Const':
            continue
        data = layer.find('data')
        constants.append({'id': layer.get('id'), 'name': layer.get('name'), **data.attrib})
    result['models'].append({'id': 'OpenVINO/' + model, 'local_dir': str(root), 'files': files,
                             'operation_counts': dict(ops), 'constant_count': len(constants),
                             'rt_info': ET.tostring(graph.find('rt_info'), encoding='unicode') if graph.find('rt_info') is not None else None})
lock = json.loads((RELEASE / 'RC4-LOCAL-INPUTS.lock.json').read_text(encoding='utf-8-sig'))
for row in lock['variants']['off']['inputs']:
    path = Path(row['source'])
    if path.name not in ('openvino.dll', 'openvino_genai.dll', 'openvino_tokenizers.dll', 'openvino_intel_gpu_plugin.dll'):
        continue
    digest = sha(path)
    result['runtime'].append({**row, 'actual_sha256': digest, 'matches_rc4_lock': digest == row['sha256']})
result['gates'] = {'source_ref': 'PASS', 'model_inventory': 'PASS',
                   'fresh_build': 'NOT_RUN', 'baseline_cpp_tests': 'NOT_RUN',
                   'strict_parser': 'NOT_RUN', 'template_round_trip': 'NOT_RUN',
                   'reasoning': 'NOT_RUN', 'unary_sse': 'NOT_RUN',
                   'cpu_load': 'NOT_RUN', 'gpu_load': 'NOT_RUN',
                   'tool_cycle_continuation': 'NOT_RUN', 'arc140v_benchmark': 'NOT_RUN',
                   'cib2_mapping': 'INCOMPLETE', 'offload': 'DISABLED', 'acceptance': 'NO'}
(OUT / 'PREFLIGHT.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'source_head': result['source_head'], 'models': [{
    'id': m['id'], 'revisions': sorted(set(f['download_revision'] for f in m['files'] if f['download_revision'])),
    'files': len(m['files']), 'operation_counts': m['operation_counts']} for m in result['models']],
    'runtime_hashes_match': all(r['matches_rc4_lock'] for r in result['runtime'])}, indent=2))
