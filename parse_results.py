import os
import json
import pandas as pd

TASKS = ['ko_nubench_symbol', 'ko_nubench_cloze', 'arc_easy', 'arc_challenge', 'winogrande', 'hellaswag']

def parse_results(root_dir):

    results = {}
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            # find the json files which starts with 'results'
            if filename.startswith("results") and filename.endswith(".json"):
                file_path = os.path.join(dirpath, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)

                    model = data['model_name_sanitized']

                    if not (model in results):
                        results[model] = {task : {'acc': 0.0, 'acc_norm': 0} for task in TASKS}

                    tasks = data['results'].keys()
                    for task in tasks:
                        acc = data['results'][task].get('acc,none')
                        acc_norm = data['results'][task].get('acc_norm,none')

                        results[model][task]['acc'] = acc
                        results[model][task]['acc_norm'] = acc_norm


                except Exception as e:
                    print(f"failed to read file: {file_path} → {e}")

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def make_csv(results):
    rows = {}
    for model, task_dict in results.items():
        row = {}
        for task, metrics in task_dict.items():
            row[(task, "acc")] = metrics.get("acc")
            row[(task, "acc_norm")] = metrics.get("acc_norm")
        rows[model] = row

    df = pd.DataFrame.from_dict(rows, orient="index")

    # 열 정렬(선택): 태스크명, 그 안에서 acc → acc_norm 순
    df = df.reindex(
        columns=sorted(df.columns, key=lambda x: (x[0], 0 if x[1]=="acc" else 1))
    )

    df.index.name = "model"
    df = df.sort_index()

    # 표시용 결측치 공란
    df_out = df.copy().where(pd.notnull(df), None)

    # ----- CSV로 저장 -----
    df_out.to_csv("results_table.csv", encoding="utf-8-sig", float_format="%.6f")

    # ----- 엑셀로 저장 (시트/고정틀 등 옵션) -----
    with pd.ExcelWriter("results_table.xlsx", engine="xlsxwriter") as writer:
        df_out.to_excel(writer, sheet_name="results")
        ws = writer.sheets["results"]
        # 헤더가 2줄(MultiIndex)이므로 2행+1열 고정(보기 편함)
        ws.freeze_panes(2, 1)

if __name__ == "__main__":
    root_dir = "/mnt/sm/KoNUBench/baseline"
    results = parse_results(root_dir=root_dir)
    make_csv(results=results)

