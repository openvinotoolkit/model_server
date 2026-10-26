"""Build explicit serving overrides, leaving downloaded export templates untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

repo = Path(__file__).resolve().parents[3]
ap = argparse.ArgumentParser()
ap.add_argument('--update-examples', action='store_true')
args = ap.parse_args()
folder = repo / 'docs/lfm/templates'
folder.mkdir(exist_ok=True)
receipt = []
for model, family in [('LFM2-24B-A2B-int4-ov', 'lfm2'), ('LFM2.5-8B-A1B-int4-ov', 'lfm25')]:
    if family == 'lfm2':
        original = subprocess.check_output(['git', '-C', str(repo), 'show', 'a5f60ccd:extras/chat_template_examples/chat_template_lfm2.jinja'], text=True)
        source = 'RC4 existing LFM2 tool-history adaptation (original export has no renderer)'
    else:
        original = (Path(r'C:\llm\models') / model / 'chat_template.jinja').read_text(encoding='utf-8')
        source = 'Unmodified target export chat_template.jinja'
    begin = original.index('{%- macro format_arg_value(')
    end = original.index('{%- endmacro -%}', begin) + len('{%- endmacro -%}')
    replacement = '{%- macro format_arg_value(arg_value) -%}\n    {{- arg_value | tojson -}}\n{%- endmacro -%}'
    updated = original[:begin] + replacement + original[end:]
    destination = folder / (model + '.jinja')
    destination.write_text(updated, encoding='utf-8', newline='\n')
    if args.update_examples:
        for base in ('extras/chat_template_examples', 'src/test/llm/chat_templates'):
            (repo / base / ('chat_template_' + family + '.jinja')).write_text(updated, encoding='utf-8', newline='\n')
    receipt.append({'model': 'OpenVINO/' + model, 'source': source,
                    'normalized_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
                    'override': destination.relative_to(repo).as_posix(), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()})
(folder / 'OVERRIDES.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
