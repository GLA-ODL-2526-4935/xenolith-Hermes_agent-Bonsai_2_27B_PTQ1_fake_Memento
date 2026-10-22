"""Render CPU tuning plans only. Never start a process or open model weights."""
import argparse
import json
from pathlib import Path

from backends.prism import load_profile, make_plan


def plans(base, engine, model):
    if base['device'] != 'none':
        raise ValueError('CPU sweep requires device none')
    result = []
    for threads in (2, 4, 6, 8):
        for batch, ubatch in ((128, 64), (256, 128)):
            for mode, count, adaptive in [('off', 0, False), ('fixed3', 3, False), ('adaptive3', 3, True)]:
                c = {**base, 'id': f'cpu-t{threads}-b{batch}-{mode}',
                     'threads': threads, 'threads_batch': threads,
                     'batch': batch, 'ubatch': ubatch, 'mtp_tokens': count,
                     'adaptive_mtp': adaptive}
                result.append({'configuration': c, 'plan': make_plan(c, engine, model)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', type=Path, default=Path(__file__).resolve().parent / 'profiles/bonsai27-linux-cpu-candidate.json')
    parser.add_argument('--engine', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    print(json.dumps(plans(load_profile(args.profile), args.engine, args.model), indent=2))


if __name__ == '__main__':
    main()
