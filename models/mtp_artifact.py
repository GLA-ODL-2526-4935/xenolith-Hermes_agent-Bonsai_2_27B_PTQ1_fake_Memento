"""Package and reconstruct exact GGUF MTP replacements using Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import zipfile


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gguf_layout(path):
    """Read tensor spans, including alignment padding; never load weight arrays."""
    with Path(path).open('rb') as f:
        def number(fmt):
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        def string():
            return f.read(number('Q')).decode('utf-8')

        def value(kind):
            formats = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
                       6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
            if kind == 8:
                string()
            elif kind == 9:
                element, count = number('I'), number('Q')
                if element in formats:
                    f.seek(count * struct.calcsize('<' + formats[element]), 1)
                else:
                    for _ in range(count):
                        value(element)
            elif kind in formats:
                return number(formats[kind])
            else:
                raise ValueError('Unsupported GGUF metadata type')

        if f.read(4) != b'GGUF' or number('I') != 3:
            raise ValueError('Expected GGUF v3')
        count, fields = number('Q'), number('Q')
        alignment = 32
        for _ in range(fields):
            name, kind = string(), number('I')
            result = value(kind)
            if name == 'general.alignment':
                alignment = result
        tensors = []
        for _ in range(count):
            name, dimensions = string(), number('I')
            shape = [number('Q') for _ in range(dimensions)]
            typ, offset = number('I'), number('Q')
            tensors.append({'name': name, 'shape': shape, 'type': typ, 'relative': offset})
        start = (f.tell() + alignment - 1) // alignment * alignment
    tensors.sort(key=lambda row: row['relative'])
    for i, row in enumerate(tensors):
        row['offset'] = start + row.pop('relative')
        end = start + tensors[i + 1]['relative'] if i + 1 < len(tensors) else Path(path).stat().st_size
        row['bytes'] = end - row['offset']
        if row['bytes'] <= 0:
            raise ValueError('Invalid tensor span')
    return start, tensors


def transfer(source, destination, count):
    while count:
        chunk = source.read(min(count, 4 * 1024 * 1024))
        if not chunk:
            raise ValueError('Truncated input')
        destination.write(chunk)
        count -= len(chunk)


def pack(base, trained, output):
    if output.exists():
        raise ValueError('Refusing to overwrite package')
    _, old = gguf_layout(base)
    header, new = gguf_layout(trained)
    lookup = {row['name']: row for row in old}
    if set(lookup) != {row['name'] for row in new}:
        raise ValueError('Tensor sets differ')
    manifest = {'format': 'bonsai-mtp-replacement-v1', 'base_sha256': digest(base),
                'output_sha256': digest(trained), 'output_bytes': trained.stat().st_size,
                'tensors': []}
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise ValueError('Refusing to overwrite an existing partial package')
    try:
        with base.open('rb') as src, trained.open('rb') as target, zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_STORED) as archive:
            with archive.open('header.bin', 'w') as dest:
                transfer(target, dest, header)
            for i, row in enumerate(new):
                previous = lookup[row['name']]
                target.seek(row['offset'])
                entry = dict(row)
                if row['name'].startswith('blk.64.'):
                    entry['member'] = f'head/{i}.bin'
                    with archive.open(entry['member'], 'w') as dest:
                        transfer(target, dest, row['bytes'])
                else:
                    if any(previous[key] != row[key] for key in ('shape', 'type', 'bytes')):
                        raise ValueError('Backbone layout differs: ' + row['name'])
                    src.seek(previous['offset'])
                    remaining = row['bytes']
                    while remaining:
                        size = min(remaining, 4 * 1024 * 1024)
                        if src.read(size) != target.read(size):
                            raise ValueError('Backbone changed: ' + row['name'])
                        remaining -= size
                    entry['base_offset'] = previous['offset']
                manifest['tensors'].append(entry)
            archive.writestr('manifest.json', json.dumps(manifest, indent=2))
        partial.rename(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def apply(base, package, output):
    if output.exists() or output.resolve() == base.resolve():
        raise ValueError('Output must be a new file; the original base is preserved')
    partial = output.with_name(output.name + '.partial')
    if partial.exists():
        raise ValueError('Refusing to overwrite an existing partial model')
    try:
        with zipfile.ZipFile(package) as archive:
            m = json.loads(archive.read('manifest.json'))
            if m['format'] != 'bonsai-mtp-replacement-v1' or digest(base) != m['base_sha256']:
                raise ValueError('Package format or base SHA256 mismatch')
            with base.open('rb') as src, partial.open('xb') as dest:
                dest.truncate(m['output_bytes'])
                with archive.open('header.bin') as header:
                    shutil.copyfileobj(header, dest, 4 * 1024 * 1024)
                for row in m['tensors']:
                    if row['offset'] < dest.tell() or row['offset'] + row['bytes'] > m['output_bytes']:
                        raise ValueError('Invalid output range')
                    dest.seek(row['offset'])
                    if 'member' in row:
                        with archive.open(row['member']) as head:
                            transfer(head, dest, row['bytes'])
                    else:
                        src.seek(row['base_offset'])
                        transfer(src, dest, row['bytes'])
            if digest(partial) != m['output_sha256']:
                raise ValueError('Reconstructed model SHA256 mismatch')
        partial.rename(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('pack')
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--trained', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('apply')
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--package', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'pack':
        pack(args.base, args.trained, args.output)
    else:
        apply(args.base, args.package, args.output)
    print('Verified:', args.output, digest(args.output), flush=True)


if __name__ == '__main__':
    main()
