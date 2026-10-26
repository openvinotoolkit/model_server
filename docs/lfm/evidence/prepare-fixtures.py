"""Copy tokenizer-only fixtures into this isolated checkout and record exact origins."""
import hashlib
import json
from pathlib import Path
import shutil

repo = Path(__file__).resolve().parents[3]
original = Path(r'C:\gm\qwenmonster\semantic-refit-2026.4-src\src\test\llm_testing')
fixtures = repo / 'src/test/llm_testing'
names = ('openvino_tokenizer.xml', 'openvino_tokenizer.bin', 'openvino_detokenizer.xml',
         'openvino_detokenizer.bin', 'tokenizer_config.json', 'chat_template.jinja')
sources = [(Path(r'C:\llm\models\LFM2-24B-A2B-int4-ov'), 'LiquidAI/LFM2-2.6B', 'target 24B tokenizer at legacy suite path; not a 2.6B model claim'),
           (Path(r'C:\llm\models\LFM2.5-8B-A1B-int4-ov'), 'LiquidAI/LFM2.5-8B-A1B', 'target export tokenizer'),
           (original / 'facebook/opt-125m', 'facebook/opt-125m', 'existing genuine template fixture'),
           (original / 'Qwen/Qwen3-8B', 'Qwen/Qwen3-8B', 'existing Qwen regression fixture'),
           (original / 'OpenVINO/gemma-4-E4B-it-int4-ov', 'OpenVINO/gemma-4-E4B-it-int4-ov', 'existing Gemma regression fixture')]
rows = []
for source, relative, note in sources:
    dest = fixtures / relative
    dest.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = source / name
        if not path.exists(): continue
        shutil.copyfile(path, dest / name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == hashlib.sha256((dest / name).read_bytes()).hexdigest()
        rows.append({'source': str(path), 'fixture': str(dest / name), 'sha256': digest, 'note': note})
(Path(__file__).parent / 'FIXTURES.json').write_text(json.dumps(rows, indent=2) + '\n')
print('Tokenizer fixture files copied and verified:', len(rows))
