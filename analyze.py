import json
import os
import sys
import pandas as pd
# data structure
# {
#     "doc_id": 0, 
#     "doc": 
#         {
#             "idx": "2270", 
#             "raw_id": 613930, 
#             "original_sentence": "그는 평양요한학교에 입학하여 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.", 
#             "standard_negation": "그는 평양요한학교에 입학하여 동요를 작곡하지 않았거나, 어린이 음악단체를 만들어 방송에 출연하지 못했다.", 
#             "sn_절1": "안 장형", 
#             "sn_절2": "못 장형", 
#             "sn_절3": null, 
#             "local_negation_type": "부사절", 
#             "local_negation": "그는 평양요한학교에 입학하지 않아 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.", "local_부정요소": "평양요한학교에 입학하여", 
#             "ln_절1": "안 장형", 
#             "ln_절2": null, 
#             "contradiction": "그는 평양요한학교에 입학이 거부되어 동요를 작곡했으며, 어린이 음악단체를 해산하여 방송에서 퇴출되었다.", 
#             "paraphrase": "평양요한학교에 다니면서 동요를 만들었고, 어린이 음악단체를 조직해 방송에도 나왔다.", 
#             "note": null, 
#             "query": "문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문: 그는 평양요한학교에 입학하여 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.\n부정문:", 
#             "choices": ["그는 평양요한학교에 입학하여 동요를 작곡하지 않았거나, 어린이 음악단체를 만들어 방송에 출연하지 못했다.", "그는 평양요한학교에 입학하지 않아 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.", "그는 평양요한학교에 입학이 거부되어 동요를 작곡했으며, 어린이 음악단체를 해산하여 방송에서 퇴출되었다.", "평양요한학교에 다니면서 동요를 만들었고, 어린이 음악단체를 조직해 방송에도 나왔다."], 
#             "gold": 0}, 
#     "target": "0", 
#     "arguments": {"gen_args_0": {"arg_0": "문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문: 그는 평양요한학교에 입학하여 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.\n부정문:", "arg_1": " 그는 평양요한학교에 입학하여 동요를 작곡하지 않았거나, 어린이 음악단체를 만들어 방송에 출연하지 못했다."}, "gen_args_1": {"arg_0": "문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문: 그는 평양요한학교에 입학하여 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.\n부정문:", "arg_1": " 그는 평양요한학교에 입학하지 않아 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다."}, "gen_args_2": {"arg_0": "문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문: 그는 평양요한학교에 입학하여 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.\n부정문:", "arg_1": " 그는 평양요한학교에 입학이 거부되어 동요를 작곡했으며, 어린이 음악단체를 해산하여 방송에서 퇴출되었다."}, "gen_args_3": {"arg_0": "문제: 다음의 원문을 standard negation을 활용하여 올바르게 부정하시오.\n\nstandard negation:\n- 주절의 서술어를 한국어의 부정 표현을 활용해 부정함으로써 원문 P를  ¬P로 만든다.\n- 주절의 서술어 외의 나머지 부분은 수정하지 않는다.\n- 원문이 조건문일 때는 논리적 규칙(¬(P → Q) ≡ P ∧ ¬Q)을 따라 부정한다.\n- 주절이 여러 개일 경우 드모르간의 법칙(예. ¬(P ∧ Q) ≡ ¬P ∨ ¬Q)에 따라 모든 주절의 서술어를 부정한다.\n\n한국어의 부정 표현:\n- 안 계열: 안, -지 않다\n- 못 계열: 못, -지 못하다\n- 어휘적 부정: 상보 반의어를 활용한 부정(이다/아니다, 있다/없다, 참석하다/불참하다 등)\n\n원문: 그는 평양요한학교에 입학하여 동요를 작곡했으며, 어린이 음악단체를 만들어 방송에 출연하였다.\n부정문:", "arg_1": " 평양요한학교에 다니면서 동요를 만들었고, 어린이 음악단체를 조직해 방송에도 나왔다."}}, 
#     "resps": [[["-25.625", "False"]], [["-23.5", "False"]], [["-65.0", "False"]], [["-71.5", "False"]]], 
#     "filtered_resps": [["-25.625", "False"], ["-23.5", "False"], ["-65.0", "False"], ["-71.5", "False"]], 
#     "doc_hash": "d94881c124647764e4a6341efd16c17b39e438c7452b669f72b1dae1395671f7", 
#     "prompt_hash": "16abd5facf18f442f591b1615a3a006649473bef0d6d19f138cb37d89f365697", 
#     "target_hash": "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9", 
#     "acc": 0.0, 
#     "acc_norm": 0.0
# }

SEED = [1234, 308, 1028]
SHOT = [0, 1, 2, 5, 10]
FEWSHOT_LABELS = [f"{x}shot" for x in SHOT] 
METHOD = ["cloze", "symbol"]
CATEGORIES = ['standard_negation', 'local_negation', 'contradiction', 'paraphrase']

def remove_prefix_suffix(filename: str):
    name = filename.removeprefix("results_").removesuffix(".json")
    return name

def _read_samples_symbol(sample_data):
    temp = {}

    if sample_data["doc"]["standard_negation"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["standard_negation"])
        if idx != -1:
            temp[idx] = "standard_negation"

    if sample_data["doc"]["local_negation"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["local_negation"])
        if idx != -1:
            temp[idx] = "local_negation"

    if sample_data["doc"]["contradiction"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["contradiction"])
        if idx != -1:
            temp[idx] = "contradiction"

    if sample_data["doc"]["paraphrase"]:
        idx = sample_data["doc"]["query"].find(sample_data["doc"]["paraphrase"])
        if idx != -1:
            temp[idx] = "paraphrase"

    # Sort by appearance order in the query (A, B, C, D)
    sorted_items = sorted(temp.items(), key=lambda x: x[0])

    # Extract mapping of choices in order (A→..., B→..., C→..., D→...)
    mapping = [item[1] for item in sorted_items]

    # Extract model scores
    resps = [float(resp[0][0]) for resp in sample_data["resps"]]

    # Identify the index of the highest-scoring option (0~3)
    chosen_idx = resps.index(max(resps))
    
    return mapping[chosen_idx]

def _read_samples_cloze(sample_data):
    choices = ['standard_negation', 'local_negation', 'contradiction', 'paraphrase']
    resps = [float(resp[0][0]) for resp in sample_data["resps"]]

    return choices[resps.index(max(resps))]

def analyze(root_dir: str, method: str, fewshot: int):
    
    # Initialize counters for each choice type
    if fewshot == 0:
        result = {
            model_name: {
                'standard_negation': 0, 
                'local_negation': 0, 
                'contradiction': 0, 
                'paraphrase': 0}
            for model_name in os.listdir(f'{root_dir}/{fewshot}shot')
        }
    else:
        result = {
            model_name: 
                {seed:
                {'standard_negation': 0, 
                    'local_negation': 0, 
                    'contradiction': 0, 
                    'paraphrase': 0} for seed in SEED}
                
                for model_name in os.listdir(f'{root_dir}/{fewshot}shot')
                }

    
    for model_name in os.listdir(f'{root_dir}/{fewshot}shot'):
        base = f'{root_dir}/{fewshot}shot/{model_name}'
        for dirpath, dirnames, filenames in os.walk(base):
            for filename in filenames:
                # find the json files which starts with 'results'
                if filename.startswith("results") and filename.endswith(".json"):
                    result_file_path = os.path.join(dirpath, filename)
        
                    try:
                        with open(result_file_path, "r", encoding="utf-8") as r:
                            result_data = json.load(r)
                        
                        seed = result_data["config"]["random_seed"]
                        name = remove_prefix_suffix(filename)

                        sample_name = f"samples_ko_nubench_{method}_{name}.jsonl"
                        sample_file_path = os.path.join(dirpath, sample_name)
                        
                        try:
                            with open(sample_file_path, "r", encoding="utf-8") as s:
                                for line in s:
                                    if not line.strip():
                                        continue

                                    sample_data = json.loads(line)
                                    if method == "symbol":
                                        chosen = _read_samples_symbol(sample_data)
                                    elif method == "cloze":
                                        chosen = _read_samples_cloze(sample_data)
                                    
                                    # Increase count for the predicted choice type
                                    if fewshot == 0:
                                        result[model_name][chosen] += 1
                                    else:
                                        result[model_name][seed][chosen] += 1

                        except Exception as e_s:
                            print(f"failed to read sample file: {sample_file_path} → {e_s}")  
                                        
                    except Exception as e_r:
                        print(f"failed to read result file: {result_file_path} → {e_r}")
                

    return result

def analyze_cloze(data_path: str):
    result = {'standard_negation': 0, 'local_negation': 0, 'contradiction': 0, 'paraphrase': 0}
    choices = ['standard_negation', 'local_negation', 'contradiction', 'paraphrase']

    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line) 
            resps = [float(resp[0][0]) for resp in data["resps"]]
            result[choices[resps.index(max(resps))]] += 1


    return result

def merge_fewshot(agg: dict, shot: int, src: dict) -> dict:
    shot_label = f"{shot}shot"
    for model, seeds in src.items():
        m = agg.setdefault(model, {})
        s = m.setdefault(shot_label, {})
        for seed, counts in seeds.items():
            s[seed] = counts
    return agg

def save_zeroshot_csv(method_name: str, zero_dict: dict, env: str):
    """
    zero_dict: zero_shot_results[method] 구조
        {
          "modelA": {"standard_negation": int, "local_negation": int, "contradiction": int, "paraphrase": int},
          "modelB": {...},
          ...
        }
    CSV: 행=model_name, 열=Z_CATEGORIES
    """
    if not zero_dict:
        print(f"[WARN] No zeroshot data for method={method_name}")
        return

    rows = []
    idx = []

    for model, counts in zero_dict.items():
        row = [int(counts.get(cat, 0)) for cat in CATEGORIES]
        rows.append(row)
        idx.append(model)

    df = pd.DataFrame(rows, index=idx, columns=CATEGORIES).sort_index()

    out_csv = f"analyze/analyze_{env}_{method_name}_0shot.csv"
    df.to_csv(out_csv, encoding='utf-8-sig')
    print(f"Saved CSV: {out_csv}, shape={df.shape}")

def save_fewshot_csv(method_name: str, merged_dict: dict, env: str):
    """
    merged_dict: merged_fewshots[method] 구조
        {
            "modelA": {
                "1shot": {seed: {4개 category}}, 
                "2shot": ...
            },
            "modelB": {...}
        }
    """
    # ---- 컬럼 MultiIndex 생성 (seed → category) ----
    col_tuples = []
    for seed in SEED:
        for cat in CATEGORIES:
            col_tuples.append((seed, cat))
    columns = pd.MultiIndex.from_tuples(col_tuples, names=['seed', 'type'])

    row_index = []
    row_data = []

    # ---- 행 생성 (model → fewshot) ----
    for model, shots in merged_dict.items():
        for shot in SHOT:
            shot_label = f"{shot}shot"
            seed_dict = shots.get(shot_label, {}) or {}

            row_vals = []
            for seed in SEED:
                counts = seed_dict.get(seed, {}) or {}
                for cat in CATEGORIES:
                    row_vals.append(int(counts.get(cat, 0)))

            row_index.append((model, shot_label))
            row_data.append(row_vals)

    # ---- 데이터프레임 생성 ----
    if not row_data:
        print(f"[WARN] No data to save for method={method_name}")
        return

    df = pd.DataFrame(
        row_data,
        index=pd.MultiIndex.from_tuples(row_index, names=['model', 'fewshot']),
        columns=columns
    )

    # ---- ⭐ 중요: fewshot 레벨을 숫자순으로 강제 정렬 ----
    df = df.reindex(
        pd.MultiIndex.from_product(
            [sorted(df.index.levels[0]), FEWSHOT_LABELS],   # model은 알파벳 순, fewshot은 숫자 순
            names=['model','fewshot']
        )
    )

    # 행 중 실제로 없는 값은 drop할 수도 있음 (옵션)
    df = df.dropna(how='all')

    # ---- CSV 저장 ----
    out_csv = f"analyze/analyze_{env}_{method_name}_fewshot.csv"
    df.to_csv(out_csv, encoding='utf-8-sig')
    print(f"Saved CSV: {out_csv}, shape={df.shape}")


if __name__ == "__main__":
    env = sys.argv[1]
    if sys.argv[1] == "gsds":
        root_dir = '/shared/erc/lab08/korean_negation/gsds_baseline'
    elif sys.argv[1] == "amd":
        root_dir = '/mnt/sm/KoNUBench/baseline'
    else:
        raise ValueError("env must be 'gsds' or 'amd'")

    # method별 산출물:
    #   - zero_shot_results[method]: model -> counts (그대로)
    #   - merged_fewshots[method]:   model -> fewshot -> seed -> counts
    zero_shot_results = {}
    merged_fewshots = {}

    for method in METHOD:
        zero_shot_results[method] = {}
        merged_fewshots[method] = {}

        for shot in SHOT:
            res = analyze(root_dir=root_dir, method=method, fewshot=shot)

            if shot == 0:
                # 0shot은 result를 그대로 둔다 (model -> counts)
                zero_shot_results[method] = res
            else:
                # fewshot은 같은 method 안에서 병합: model -> "{shot}shot" -> seed -> counts
                merged_fewshots[method] = merge_fewshot(merged_fewshots[method], shot, res)

    # 필요하면 파일로도 저장
    
    for method in METHOD:
        zero_out_name = f"analyze/analyze_{env}_{method}_0shot.json"
        few_out_name = f"analyze/analyze_{env}_{method}_fewshot.json"

        with open(zero_out_name, "w", encoding="utf-8") as f:
            json.dump(zero_shot_results[method], f, ensure_ascii=False, indent=2)
            save_zeroshot_csv(method, zero_shot_results[method], env)
        with open(few_out_name, "w", encoding="utf-8") as f:
            json.dump(merged_fewshots[method], f, ensure_ascii=False, indent=2)
            save_fewshot_csv(method, merged_fewshots[method], env)