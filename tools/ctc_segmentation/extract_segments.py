import argparse
import json
import os
from pathlib import Path
from random import randint

from nemo.utils import logging


def get_samples(manifest, n=4):
    samples = []
    with open(manifest, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # first n samples
    samples.extend(lines[:n])

    # random in the middle
    for i in range(n):
        samples.append(lines[randint(n, len(lines) - n)])
    # last n samples
    samples.extend(lines[-n:])
    return samples


parser = argparse.ArgumentParser(description="Extract samples from each manifest")
parser.add_argument(
    "--num_samples",
    default=3,
    type=int,
    help='The number of samples to get from the beginning, end and random fromt he middle',
)
parser.add_argument(
    "--manifests_dir",
    default='_high_score_manifest.json',
    type=str,
    help='Path to directory with *_high_score_manifest.json files',
)
parser.add_argument("--output_dir", default='output', help='Path to the directory to store final manifest')

if __name__ == '__main__':
    args = parser.parse_args()
    manifest_dir = Path(args.manifests_dir)
    manifests = manifest_dir.glob('*_high_score_manifest.json')
    sample_manifest = os.path.join(args.output_dir, 'sample_manifest.json')

    with open(sample_manifest, 'w', encoding='utf8') as f:
        for manifest in sorted(list(manifests)):
            samples = get_samples(manifest, args.num_samples)
            for sample in samples:
                f.write(sample)

    print(f'Done. Samples are saved at {sample_manifest}.')
