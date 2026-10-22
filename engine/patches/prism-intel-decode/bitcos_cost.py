"""Read-only, bounded-memory PTQ1 GGUF packing feasibility. Never rewrites a model.

--model explicitly performs a potentially long disk scan; no default model/path.
Sampling is labelled an estimate and cannot select a production representation.
"""
import argparse
import json
from pathlib import Path
import struct

PTQ1_TYPE = 143  # pinned Prism enum, not an upstream/general ternary type
SIZES = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}


def _zero_table(width):
    table=[]
    for value in range(256):
        zeros=0
        for _ in range(width):
            zeros+=(value*3 >> 8)==1
            value=value*3 & 255
        table.append(zeros)
    return table


ZERO5, ZERO4 = _zero_table(5), _zero_table(4)


def trits(block):
    if len(block) != 28:
        raise ValueError('PTQ1 block requires exactly 28 bytes')
    out = []
    for position in range(128):
        if position < 80:
            value, times = block[position % 16], position // 16
        elif position < 120:
            value, times = block[16+(position-80) % 8], (position-80)//8
        else:
            value, times = block[24+(position-120) % 2], (position-120)//2
        for _ in range(times):
            value = value*3 & 255
        out.append((value*3 >> 8)-1)
    return out


def block_bytes(block):
    if len(block)!=28:
        raise ValueError('PTQ1 block requires exactly 28 bytes')
    zeros=sum(ZERO5[b] for b in block[:24])+ZERO4[block[24]]+ZERO4[block[25]]
    return 18+(128-zeros+7)//8, zeros


class Reader:
    def __init__(self, stream):
        self.f = stream
        self.size = stream.seek(0, 2)
        stream.seek(0)

    def read(self, n):
        if n < 0 or n > self.size-self.f.tell():
            raise ValueError('Truncated or oversized GGUF field')
        data = self.f.read(n)
        if len(data) != n:
            raise ValueError('Truncated GGUF')
        return data

    def skip(self, n):
        if n < 0 or n > self.size-self.f.tell():
            raise ValueError('GGUF field exceeds file')
        self.f.seek(n, 1)

    def number(self, fmt):
        return struct.unpack('<'+fmt, self.read(struct.calcsize('<'+fmt)))[0]

    def string(self):
        n = self.number('Q')
        if n > 1 << 20:
            raise ValueError('GGUF identifier exceeds bounded reader limit')
        return self.read(n).decode('utf-8')

    def skip_value(self, kind, depth=0):
        if depth > 4:
            raise ValueError('Nested metadata array depth exceeds reader limit')
        if kind in SIZES:
            self.skip(SIZES[kind])
        elif kind == 8:
            self.skip(self.number('Q'))
        elif kind == 9:
            item, count = self.number('I'), self.number('Q')
            if item in SIZES:
                self.skip(count*SIZES[item])
            else:
                if count > self.size-self.f.tell():
                    raise ValueError('Metadata array exceeds file')
                for _ in range(count):
                    self.skip_value(item, depth+1)
        else:
            raise ValueError('Unsupported GGUF metadata type')


def inventory(stream):
    r = Reader(stream)
    if r.read(4) != b'GGUF' or r.number('I') not in (2, 3):
        raise ValueError('Expected little-endian GGUF v2/v3')
    tensors, metadata = r.number('Q'), r.number('Q')
    if tensors > 100000 or metadata > 100000:
        raise ValueError('GGUF inventory exceeds bounded reader limit')
    alignment = 32
    for _ in range(metadata):
        key, kind = r.string(), r.number('I')
        if key == 'general.alignment':
            if kind != 4:
                raise ValueError('Alignment must be UINT32')
            alignment = r.number('I')
            if alignment < 1 or alignment > 1048576 or alignment & (alignment-1):
                raise ValueError('Invalid GGUF alignment')
        else:
            r.skip_value(kind)
    selected = []
    names = set()
    for _ in range(tensors):
        name, dims = r.string(), r.number('I')
        if name in names or dims < 1 or dims > 4:
            raise ValueError('Duplicate tensor or invalid dimensions')
        names.add(name)
        shape = [r.number('Q') for _ in range(dims)]
        kind, offset = r.number('I'), r.number('Q')
        if kind == PTQ1_TYPE:
            if not all(shape) or shape[0] % 128 or offset % alignment:
                raise ValueError('PTQ1 tensor shape/offset violates pinned layout')
            blocks = shape[0]//128
            rows = 1
            for d in shape[1:]:
                rows *= d
            selected.append(dict(name=name, shape=shape, offset=offset, blocks_per_row=blocks, rows=rows))
    base = (stream.tell()+alignment-1)//alignment*alignment
    intervals = []
    for t in selected:
        t['start'] = base+t['offset']
        t['raw_bytes'] = t['rows']*t['blocks_per_row']*28
        if t['start'] > r.size or t['raw_bytes'] > r.size-t['start']:
            raise ValueError('PTQ1 tensor exceeds file')
        intervals.append((t['start'], t['start']+t['raw_bytes']))
    intervals.sort()
    if any(a[1] > b[0] for a,b in zip(intervals, intervals[1:])):
        raise ValueError('Overlapping PTQ1 tensors')
    return selected


def analyze(path, max_blocks=None):
    if max_blocks is not None and (type(max_blocks) is not int or max_blocks < 1):
        raise ValueError('Sample limit must be positive')
    results = []
    with Path(path).open('rb') as f:
        for t in inventory(f):
            total = t['rows']*t['blocks_per_row']
            sampled = min(total, max_blocks) if max_blocks else total
            payload = zeros = 0
            # Evenly spaced deterministic samples across the entire tensor.
            # Full scans read sequentially without an expanded model copy.
            f.seek(t['start'])
            for i in range(sampled):
                index = i*total//sampled
                if sampled != total:
                    f.seek(t['start']+28*index)
                block = f.read(28)
                size, z = block_bytes(block)
                payload += size
                zeros += z
            estimated_payload = payload*total/sampled
            # Row-local offsets include the end sentinel. Small per-tensor
            # descriptors/alignment are not included, so this is still a floor.
            indices = t['rows']*(t['blocks_per_row']+1)*4
            converted = estimated_payload+indices
            exact = sampled == total
            representable = t['blocks_per_row']*34 < 2**32
            results.append({**t, 'sampled_blocks':sampled, 'total_blocks':total,
                'exact':exact, 'zero_fraction':zeros/(sampled*128),
                'bitcos_block_payload_bytes':estimated_payload,
                'row_offset_bytes':indices, 'bitcos_bytes_floor':converted,
                'row_offsets_representable':representable,
                'candidate_for_target_benchmark':exact and representable and converted < t['raw_bytes'],
                'default':'PTQ1', 'memory_win_is_not_speed_win':True})
    return {'schema_version':1, 'model':str(Path(path)), 'model_bytes':Path(path).stat().st_size,
            'model_sha256_verified':False, 'model_modified':False, 'tensors':results,
            'limits':'Packing floor excludes descriptors, page-cache and driver allocations; production adoption requires full hash, target parity and CPU/Vulkan layout support.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    group=p.add_mutually_exclusive_group()
    group.add_argument('--max-blocks-per-tensor', type=int, default=4096)
    group.add_argument('--full-scan',action='store_true',help='Read every PTQ1 block; may be slow on a large file')
    args = p.parse_args()
    print(json.dumps(analyze(args.model,None if args.full_scan else args.max_blocks_per_tensor),indent=2))


if __name__ == '__main__':
    main()
