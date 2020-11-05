import argparse
import json
import os
from pathlib import Path
from random import randint


def get_samples(manifest, num_samples):
    samples = []
    with open(manifest, 'r', encoding='utf8') as f:
        lines = f.readlines()

    # first n samples
    samples.extend(lines[:num_samples])
    # random in the middle
    for i in range(num_samples):
        samples.append(lines[randint(num_samples, len(lines) - num_samples)])
    # last n samples
    samples.extend(lines[-num_samples:])
    return samples


parser = argparse.ArgumentParser(description="Extract samples from each manifest")
parser.add_argument(
    "--manifests_dir",
    default='_high_score_manifest.json',
    type=str,
    required=True,
    help='Path to directory with *_high_score_manifest.json files',
)
parser.add_argument(
    "--output_dir", default='output', required=True, help='Path to the directory to store final manifest'
)
parser.add_argument(
    "--num_samples",
    default=3,
    type=int,
    help='The number of samples to get from the beginning, end and random fromt he middle',
)

if __name__ == '__main__':
    args = parser.parse_args()
    manifest_dir = Path(args.manifests_dir)
    manifests = sorted(list(manifest_dir.glob('*_high_score_manifest.json')))
    sample_manifest = os.path.join(args.output_dir, 'sample_manifest.json')
    all_manifest = os.path.join(args.output_dir, 'all_manifest.json')

    with open(sample_manifest, 'w', encoding='utf8') as f:
        for manifest in manifests:
            samples = get_samples(manifest, args.num_samples)
            for sample in samples:
                f.write(sample)

    total_duration = 0
    with open(all_manifest, 'w') as out_f:
        for manifest in manifests:
            with open(manifest, 'r', encoding='utf8') as f:
                for line in f:
                    info = json.loads(line)
                    total_duration += info['duration']
                    out_f.write(line)

    print(f'Sample manifest is saved at {sample_manifest}')
    print(f'Aggregated manifest is saved at {all_manifest}')
    print(f'Total files duration: {round(total_duration/60)} min or ~{round(total_duration/60/60)} hr')
    print('Done.')
