"""Render the unmodified export macro; diagnostic only, not minja acceptance."""
import json
from pathlib import Path
import sys

sys.path.insert(0, r'C:\gm\qwenmonster\releases\gemmamonster-2026.4-rc4\extracted-py_on\ovms\python\Lib\site-packages')
from jinja2 import Environment

path = Path(r'C:\llm\models\LFM2.5-8B-A1B-int4-ov\chat_template.jinja')
source = path.read_text(encoding='utf-8')
macro = source[source.index('{%- macro format_arg_value'):source.index('{%- macro parse_content')]
render = Environment().from_string(macro).module.format_arg_value
cases = ["O'Brien", 'C:\\test', 'line\nnext', '"quoted"', '末尾\\', {'path': 'C:\\test', 'enabled': True}, ['a', "b'c"]]
rows = [{'value': value, 'export_macro_output': str(render(value)),
         'json_literal_for_comparison': json.dumps(value, ensure_ascii=False)} for value in cases]
result = {'template': str(path), 'scope': 'unmodified format_arg_value macro in Jinja2 only; not minja or parser round-trip', 'cases': rows}
Path(__file__).with_name('template-probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, ensure_ascii=True, indent=2))
