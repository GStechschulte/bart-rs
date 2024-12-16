import argparse
import glob
import pandas as pd
from pathlib import Path

def aggregate_results(results_dir):
    results = []
    files = glob.glob(f"{results_dir}/benchmark_*.txt")

    for file in files:
        filename = Path(file).stem
        params = filename.split('_')
        param1 = float(params[1].replace('p1', ''))
        param2 = float(params[2].replace('p2', ''))
        iter_num = int(params[3].replace('iter', ''))

        with open(file, 'r') as f:
            runtime = float(f.read().strip())

        results.append({
            'param1': param1,
            'param2': param2,
            'iteration': iter_num,
            'runtime': runtime
        })

    df = pd.DataFrame(results)
    summary = df.groupby(['param1', 'param2'])['runtime'].agg(['mean', 'std'])
    summary.to_csv(f"{results_dir}/summary.csv")

if __name__ == "__main__":
    results_dir = sys.argv[1]
    aggregate_results(results_dir)
