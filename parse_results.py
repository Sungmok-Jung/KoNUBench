import os
import json
import pandas as pd
from datetime import datetime

NOW = datetime.now()
TIMESTAMP = NOW.strftime("%m%d_%H%M")

ZEROSHOT_TASKS = ['ko_nubench_symbol', 'ko_nubench_cloze', 'arc_easy', 'arc_challenge', 'winogrande', 'hellaswag']
def parse_0shot_results(root_dir: str, env: str):
    results = {}
    base = os.path.join(root_dir, f"0shot")
    for dirpath, dirnames, filenames in os.walk(base):
        for filename in filenames:
            # find the json files which starts with 'results'
            if filename.startswith("results") and filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    model = data['model_name_sanitized']

                    if not (model in results):
                        results[model] = {task : {'acc': 0.0, 'acc_norm': 0.0} for task in ZEROSHOT_TASKS}

                    tasks = data['results'].keys()
                    for task in tasks:
                        acc = data['results'][task].get('acc,none')
                        acc_norm = data['results'][task].get('acc_norm,none')

                        results[model][task]['acc'] = acc
                        results[model][task]['acc_norm'] = acc_norm


                except Exception as e:
                    print(f"failed to read file: {file_path} → {e}")

    with open(f"results_0shot_{env}_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

FEWSHOT_TASKS = ['ko_nubench_symbol', 'ko_nubench_cloze']
SEED = [1234, 308, 1028]
def parse_fewshot_results(root_dir: str, fewshot: int):
    results = {}
    base = os.path.join(root_dir, f"{fewshot}shot")
    for dirpath, dirnames, filenames in os.walk(base):
        for filename in filenames:
            if filename.startswith("results") and filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    model = data['model_name_sanitized']
                    
                    # initialize
                    if not (model in results):
                        results[model] = {
                                            task: {
                                                f'{fewshot}shot': {
                                                    seed: {'acc': 0.0, 'acc_norm': 0.0}
                                                    for seed in SEED
                                                }
                                            }
                                            for task in FEWSHOT_TASKS
                                        }
                    
                    seed = data['config']['random_seed']
                    tasks = data['results'].keys()
                    for task in tasks:
                        acc = data['results'][task].get('acc,none')
                        acc_norm = data['results'][task].get('acc_norm,none')

                        results[model][task][f'{fewshot}shot'][seed]['acc'] = acc
                        results[model][task][f'{fewshot}shot'][seed]['acc_norm'] = acc_norm
                except Exception as e:
                    print(f"failed to read file: {file_path} → {e}")         
    
    # get average results
    for model, tasks in results.items():
        for task, shots_dict in tasks.items():
            shot_key = f'{fewshot}shot'
            seed_dict = shots_dict[shot_key]

            accs = []
            acc_norms = []

            for seed in SEED:
                accs.append(seed_dict[seed]['acc'])
                acc_norms.append(seed_dict[seed]['acc_norm'])

            avg_acc = sum(accs) / len(accs)
            avg_acc_norm = sum(acc_norms) / len(acc_norms)

            shots_dict[shot_key]['average'] = {
                'acc': avg_acc,
                'acc_norm': avg_acc_norm
            }

    return results

def deep_merge_fewshot(dst, src):
    for model, tasks in src.items():
        m = dst.setdefault(model, {})
        for task, shots in tasks.items():
            t = m.setdefault(task, {})
            for shot_label, seeds_dict in shots.items():
                s = t.setdefault(shot_label, {})
                s.update(seeds_dict)

    return dst

def make_0shot_csv(results: dict, env: str):
    rows = {}
    for model, task_dict in results.items():
        row = {}
        for task, metrics in task_dict.items():
            row[(task, "acc")] = metrics.get("acc")
            row[(task, "acc_norm")] = metrics.get("acc_norm")
        rows[model] = row

    df = pd.DataFrame.from_dict(rows, orient="index")

    # sort the columns by task names(acc, acc_norm)
    df = df.reindex(
        columns=sorted(df.columns, key=lambda x: (x[0], 0 if x[1]=="acc" else 1))
    )

    df.index.name = "model"
    # sort the rows by model names
    df = df.sort_index()

    df_out = df.copy().where(pd.notnull(df), None)

    # ----- save to csv -----
    df_out.to_csv(f"results_0shot_{env}_{TIMESTAMP}.csv", encoding="utf-8-sig", float_format="%.6f")

def make_fewshot_csv(results: dict, env:str):
    rows = {}

    for model, task_dict in results.items():
        row = {}
        for task, shots in task_dict.items():
            for shot_key, seeds in shots.items():  # ex) '1shot', '2shot', ...
                for seed_key, metrics in seeds.items():  # ex) '1234', '308', '1028', 'average'
                    # sanity check 
                    if not isinstance(metrics, dict):
                        continue
                    for metric in ('acc', 'acc_norm'):
                        val = metrics.get(metric, None)

                        seed_col = seed_key if seed_key == 'average' else int(seed_key)
                        col = (task, shot_key, seed_col, metric)
                        row[col] = val
        rows[model] = row

    df = pd.DataFrame.from_dict(rows, orient='index')

    # ---- sort the columns( task → shots → seed(1234,308,1028,average) → metric(acc, acc_norm) ) ----
    def _col_key(col):
        task, shot_key, seed_col, metric = col
        # '10shot' → 10
        try:
            shot_num = int(str(shot_key).replace('shot', ''))
        except Exception:
            shot_num = 10**9
        # seed ordering
        seed_order_map = {1234: 0, 308: 1, 1028: 2, 'average': 3}
        seed_order = seed_order_map.get(seed_col, 99)
        metric_order = 0 if metric == 'acc' else 1
        return (task, shot_num, seed_order, metric_order)

    if len(df.columns) > 0:
        df = df.reindex(columns=sorted(df.columns, key=_col_key))

    df.index.name = 'model'
    # sort the rows by model names
    df = df.sort_index()

    df_out = df.where(pd.notnull(df), None)

    # ----- save to csv -----
    df_out.to_csv(f"results_fewshot_{env}_{TIMESTAMP}.csv", encoding='utf-8-sig', float_format='%.6f')

if __name__ == '__main__':
    gsds_root_dir = '/shared/erc/lab08/korean_negation/gsds_baseline'
    amd_root_dir = '/mnt/sm/KoNUBench/baseline'
    
    env = 'gsds'
    # env = 'amd'

    results_0shot = parse_0shot_results(root_dir=gsds_root_dir, env=env)
    make_0shot_csv(results=results_0shot, env=env)

    results_1shot = parse_fewshot_results(root_dir=gsds_root_dir, fewshot=1)
    results_2shot = parse_fewshot_results(root_dir=gsds_root_dir, fewshot=2)
    results_5shot = parse_fewshot_results(root_dir=gsds_root_dir, fewshot=5)
    results_10shot = parse_fewshot_results(root_dir=gsds_root_dir, fewshot=10)

    # results_0shot = parse_0shot_results(root_dir=amd_root_dir)
    # results_1shot = parse_fewshot_results(root_dir=amd_root_dir, fewshot=1)
    # results_2shot = parse_fewshot_results(root_dir=amd_root_dir, fewshot=2)
    # results_5shot = parse_fewshot_results(root_dir=amd_root_dir, fewshot=5)
    # results_10shot = parse_fewshot_results(root_dir=amd_root_dir, fewshot=10)

    results_fewshot = {}
    for d in (results_1shot, results_2shot, results_5shot, results_10shot):
        results_fewshot = deep_merge_fewshot(results_fewshot, d)

    with open(f"results_fewshot_{env}_{TIMESTAMP}.json", "w", encoding="utf-8") as f:
        json.dump(results_fewshot, f, ensure_ascii=False, indent=2)
    
    make_fewshot_csv(results=results_fewshot, env=env)
