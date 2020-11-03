import os
import re
from pathlib import Path

import pandas as pd


def process_log_file(log_file):
    processes = {}
    with open(log_file, 'r') as f:
        for line in f:
            if line[0].isdigit() and 'Process-' in line:
                process_id = re.search(r'Process-\d+', line)[0]
                window_size = re.findall(r'\d+', line)[-1]
                path = re.search(r'\/.*?\.[\w:]+', line)
                if 'is processing' in line:
                    if process_id not in processes:
                        processes[process_id] = {}
                    processes[process_id]['Window Increased'] = False
                    processes[process_id]['Start Window'] = window_size
                    processes[process_id]['Final Window'] = window_size
                    base_dir, base_name = os.path.split(path[0])
                    processes[process_id]['Audio Dir'] = base_dir
                    processes[process_id]['Audio File'] = base_name
                elif 'Increasing' in line:
                    processes[process_id]['Final Window'] = window_size
                    processes[process_id]['Window Increased'] = True
                elif 'completed' in line:
                    processes[process_id]['Completed'] = True

    df = pd.DataFrame.from_dict(
        processes,
        columns=['Audio Dir', 'Audio File', 'Start Window', 'Final Window', 'Window Increased', 'Completed'],
        orient='index',
    ).reset_index()
    return df


if __name__ == '__main__':
    log_dir = '/home/ebakhturina/data/segmentation/librivox/ru/01_goncharov_obryv/debug/segments/logs'
    log_files = Path(log_dir).glob('*.log')
    dfs = []
    for log in log_files:
        dfs.append(process_log_file(log))

    dfs = pd.concat(dfs)
    summary_file = os.path.join(log_dir, 'log_summary.csv')
    dfs.to_csv(summary_file, index=False)
    print(f'Log summary saved to {summary_file}')
