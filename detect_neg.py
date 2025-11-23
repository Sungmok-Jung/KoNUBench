import json
import random
from pathlib import Path
from collections import Counter

SHORT_AHN = [" 안 "]
LONG_AHN  = ["지않", "지 않", "진 않", "지는 않", "지도 않", "지조차 않", "지만은 않", "지만도 않", "지 아니", 
             "진 아니", "지는 아니", "지조차 아니", "지만은 아니", "지만도 아니"]
SHORT_MOT = [" 못 "]
LONG_MOT  = ["지못", "지 못", "진 못", "지는 못", "지도 못", "지조차 못", "지만은 못", "지만도 못"]
MALDA     = ["지말", "지 말", "진 말", "지는 말", "지도 말", "지조차 말", "지만은 말"]

def make_json_csv(start: int, end: int, total: int, corpus_type: str, seed: int):
    random.seed(seed)
    numbers = random.sample(range(start, end + 1), total)
    print("="*50 + corpus_type + "="*50)
    print(f"🎲 random seed: {seed}")
    print(f"✅ picked_numbers: {numbers}")

    output = []

    counts = Counter({"안 단형":0, "안 장형":0, "못 단형":0, "못 장형":0, "말다 부정":0, "어휘적":0})
    #without lexical negation
    counts_nonlex = Counter({"안 단형":0, "안 장형":0, "못 단형":0, "못 장형":0, "말다 부정":0})

    total_scanned = 0 
    total_scanned_nonlex = 0     #without lexical negation

    for number in numbers:
        file_path = Path(f"/shared/erc/lab08/korean_negation/{corpus_type}_data/{corpus_type}_data_{number}.json")
        if not file_path.exists():
            print(f"⚠️ 파일 없음: {file_path}")
            continue

        try:
            data_list = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            print(f"❌ JSON 읽기 실패: {file_path} ({e})")
            continue

        # data_list가 리스트인지 확인
        if not isinstance(data_list, list):
            continue

        for data in data_list:
            sentence = data.get("sentence", "")
            data_id  = data.get("data_id", "")
            idx      = data.get("idx", "")
            total_scanned += 1

            detect = {
                "file": file_path.name,
                "data_id": data_id,
                "idx": idx,
                "sentence": sentence,
                "안 단형": any(p in sentence for p in SHORT_AHN),
                "안 장형": any(p in sentence for p in LONG_AHN),
                "못 단형": any(p in sentence for p in SHORT_MOT),
                "못 장형": any(p in sentence for p in LONG_MOT),
                "말다 부정": any(p in sentence for p in MALDA)
            }

            if not any(detect[k] for k in counts.keys()):
                continue

            output.append(detect)


            for k in counts.keys():
                if detect[k]:
                    counts[k] += 1

            #without lexical negation
            if not detect["어휘적"]:
                total_scanned_nonlex += 1
                for k in counts_nonlex.keys():
                    if detect[k]:
                        counts_nonlex[k] += 1

    # save
    out_path = Path(f"/shared/erc/lab08/korean_negation/{corpus_type}_neg_start{start}_end{end}_{len(numbers)}files_seed{seed}.json")
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    kept = len(output)

    overall_ratio = (kept / total_scanned * 100) if total_scanned else 0.0

    print(f"✅ {corpus_type}: {kept}개의 문장을 저장했습니다 → {out_path}")
    print(f"📊 {corpus_type} 내 부정문 비율(전체): {overall_ratio:.2f}%  (분모={total_scanned} 문장)")

    print(f"\n📊 {corpus_type} 데이터 부정 표현 분포 (전체 분모={kept} 문장)")
    for k, v in counts.items():
        pct = (v / kept * 100) if kept else 0.0
        if k == "어휘적":
            print(f"  • {k}: {v}개, 전체 비율: {pct:.2f}%")
        else:
            #without lexical negation
            pct_nonlex = (counts_nonlex[k] / total_scanned_nonlex * 100) if total_scanned_nonlex else 0.0
            print(f"  • {k}: {v}개, 전체 비율: {pct:.2f}%, 어휘적 제외 비율: {pct_nonlex:.2f}%")

    return {
        "output": output,
        "counts": counts,
        "counts_nonlex": counts_nonlex,
        "scanned": total_scanned,
        "scanned_nonlex": total_scanned_nonlex,
    }

if __name__ == "__main__":
    seed = 2025
    total = 15

    w = make_json_csv(start=1, end=28264, total=total, corpus_type="written", seed=seed)
    print()
    s = make_json_csv(start=1, end=37790, total=total, corpus_type="spoken",  seed=seed)
    print()

    # 전체 요약
    kept_total = len(w["output"]) + len(s["output"])
    scanned_total = w["scanned"] + s["scanned"]
    overall_ratio_total = (kept_total / scanned_total * 100) if scanned_total else 0.0
    print(f"📊 구어체+문어체 전체 부정문 비율: {overall_ratio_total:.2f}%  (분자={kept_total}, 분모={scanned_total})")

    print("\n📊 유형별 합산 (전체 분모=탐지문장 합계, 어휘적 제외 분모=어휘적 아닌 문장 합계)")
    for k in ["안 단형", "안 장형", "못 단형", "못 장형", "말다 부정", "어휘적"]:
        v_total = w["counts"][k] + s["counts"][k]
        pct_total = (v_total / kept_total * 100) if kept_total else 0.0

        if k == "어휘적":
            print(f"  • {k}: {v_total}개, 전체 비율: {pct_total:.2f}%")
        else:
            nonlex_total_den = w["scanned_nonlex"] + s["scanned_nonlex"]
            #without lexical negation
            v_nonlex = w["counts_nonlex"][k] + s["counts_nonlex"][k]
            pct_nonlex_total = (v_nonlex / nonlex_total_den * 100) if nonlex_total_den else 0.0
            print(f"  • {k}: {v_total}개, 전체 비율: {pct_total:.2f}%, 어휘적 제외 비율: {pct_nonlex_total:.2f}%")