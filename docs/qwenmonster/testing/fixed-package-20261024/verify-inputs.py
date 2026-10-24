"""Verify frozen local inputs or their staged copies; never starts OVMS."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def tree_pin(root):
    paths = sorted((p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts
                    and p.suffix.lower() not in ('.pyc', '.pyo')),
                   key=lambda p: p.relative_to(root).as_posix())
    records = [f'{digest(p)}\t{p.stat().st_size}\t{p.relative_to(root).as_posix()}\n' for p in paths]
    return len(paths), sum(p.stat().st_size for p in paths), hashlib.sha256(''.join(records).encode('utf-8')).hexdigest()


def destination(root, relative):
    path = Path(relative)
    if path.is_absolute() or path.drive or '..' in path.parts or ':' in relative:
        raise ValueError('Unsafe payload destination: ' + relative)
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError('Destination escapes payload root: ' + relative)
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, default=Path(__file__).with_name('LOCAL-INPUTS.lock.json'))
    parser.add_argument('--payload-root', type=Path, help='Check staged copies instead of original absolute paths')
    args = parser.parse_args()
    lock = json.loads(args.lock.read_text(encoding='utf-8-sig'))
    errors = []
    checked = 0
    rows = list(lock['payload_files'])
    if not args.payload_root:
        rows += [lock['sdk_archive'], lock['debug_test_artifact_not_shipped']]
        rows += lock['original_launchers'] + lock['build_only']
    try:
        for row in rows:
            path = destination(args.payload_root, row['destination']) if args.payload_root else Path(row['source'])
            if not path.is_file():
                errors.append('Missing file: ' + str(path))
            elif path.stat().st_size != row['bytes'] or digest(path) != row['sha256']:
                errors.append('File mismatch: ' + str(path))
            checked += 1
        for row in lock['payload_trees']:
            root = destination(args.payload_root, row['destination']) if args.payload_root else Path(row['source'])
            if not root.is_dir():
                errors.append('Missing directory: ' + str(root))
            elif tree_pin(root) != (row['files'], row['bytes'], row['tree_sha256']):
                errors.append('Directory mismatch: ' + str(root))
            checked += 1
        for scan in lock['import_scan']:
            for dep in scan['dependencies']:
                if dep['resolution'] == 'unresolved':
                    errors.append('Unresolved captured import: ' + scan['file'] + ' -> ' + dep['name'])
    except (OSError, ValueError) as error:
        errors.append(str(error))
    print(json.dumps({'status': 'FAIL' if errors else 'PASS', 'mode': 'payload' if args.payload_root else 'local-inputs',
                      'checks': checked, 'errors': errors,
                      'scope': 'Pinned file bytes/directory trees and saved static-import report; not a fresh PE scan, dynamic-load check or runtime acceptance.'}, indent=2))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
