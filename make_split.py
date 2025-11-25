import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np

HUMAN_EVAL = [
    '5051', '3212', '4064', '286', '284', '2567', '2062', '500', '1853', '1120',
    '1976', '2371', '4478', '2440', '711', '1426', '4850', '4855', '1882', '2379',
    '3327', '702', '222', '2788', '4035', '1946', '2604', '914', '1759', '4428',
    '1868', '2972', '592', '2347', '836', '1451', '3939', '386', 'G139', '1953',
    '3035', '2464', '1522', '902', '2344', '2021', '5001', '2186', '3698', '1906'
]


def split_dataset(
    csv_path: str,
    out_dir: str,
    train_size: int = 2500,
    val_size: int = 1000,
    seed: int = 42,
):
    csv_path = Path(csv_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    if "idx" not in df.columns:
        raise KeyError("CSV에 'raw_id' 컬럼이 없습니다.")

    df["idx_str"] = df["idx"].astype(str)

    # Sanity check about human eval list
    all_ids = set(df["idx_str"])
    present_eval = sorted(set(HUMAN_EVAL) & all_ids)
    missing_eval = sorted(set(HUMAN_EVAL) - all_ids)

    print(f"전체 샘플 수: {len(df)}")
    print(f"CSV 안에서 실제로 찾은 HUMAN_EVAL ID 수: {len(present_eval)}")
    print(f"  존재하는 HUMAN_EVAL ID 예시: {present_eval[:10]}")
    print(f"  없는 HUMAN_EVAL ID 예시: {missing_eval[:10]}")

    if len(present_eval) == 0:
        raise ValueError(
            "HUMAN_EVAL 리스트에 있는 ID가 CSV의 'id'와 하나도 매칭되지 않습니다.\n"
            "id 값 형식(공백, prefix 등)을 확인해 주세요."
        )

    # HUMAN_EVAL에 해당하는 행과 나머지 행 분리 (raw_id_str 기준)
    is_human_eval = df["idx_str"].isin(HUMAN_EVAL)
    df_human = df[is_human_eval].copy()
    df_rest = df[~is_human_eval].copy()

    print(f"HUMAN_EVAL에 해당하는 샘플 수(df_human): {len(df_human)}")
    print(f"나머지(rest) 샘플 수(df_rest): {len(df_rest)}")

    # train/val은 HUMAN_EVAL이 아닌 데이터에서만 뽑음
    if len(df_rest) < train_size + val_size:
        raise ValueError(
            f"HUMAN_EVAL을 제외한 나머지 데이터가 {len(df_rest)}개라 "
            f"train({train_size}) + val({val_size}) = {train_size + val_size}개를 만들 수 없습니다."
        )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(df_rest))  # df_rest 기준 인덱스(0..len-1)

    # df_rest에서 iloc 기반으로 train/val/test_rest 나누기
    train_idx = perm[:train_size]
    val_idx = perm[train_size:train_size + val_size]
    test_rest_idx = perm[train_size + val_size:]

    train_df = df_rest.iloc[train_idx].copy()
    val_df = df_rest.iloc[val_idx].copy()
    test_rest_df = df_rest.iloc[test_rest_idx].copy()

    # test = (rest에서 남은 애들) + (HUMAN_EVAL)
    test_df = pd.concat([test_rest_df, df_human], ignore_index=True)

    # ===== 중복 검사: raw_id_str 기준으로 disjoint 여부 확인 =====
    train_ids = set(train_df["idx_str"])
    val_ids = set(val_df["idx_str"])
    test_ids = set(test_df["idx_str"])

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    if overlap_train_val:
        raise AssertionError(f"train과 val 사이에 중복 raw_id가 있습니다: {list(overlap_train_val)[:10]}")
    if overlap_train_test:
        raise AssertionError(f"train과 test 사이에 중복 raw_id가 있습니다: {list(overlap_train_test)[:10]}")
    if overlap_val_test:
        raise AssertionError(f"val과 test 사이에 중복 raw_id가 있습니다: {list(overlap_val_test)[:10]}")

    # ===== test에 HUMAN_EVAL raw_id가 모두 포함되는지 확인 =====
    missing_in_test = [hid for hid in HUMAN_EVAL if hid not in test_ids]
    if missing_in_test:
        raise AssertionError(
            f"다음 HUMAN_EVAL ID들이 test에 포함되지 않았습니다: {missing_in_test}"
        )

    print(f"train 크기: {len(train_df)}")
    print(f"val 크기: {len(val_df)}")
    print(f"test 크기: {len(test_df)} (HUMAN_EVAL 포함)")

    # JSON 저장 (list of dicts)
    def save_json(df_part: pd.DataFrame, path: Path):
        # 1) JSON에 넣지 않을 내부 컬럼 제거
        tmp = df_part.drop(columns=["idx_str"]).copy()

        # 2) 모든 컬럼을 object로 변환 (float 컬럼에서도 None이 NaN으로 되돌아가지 않게)
        for col in tmp.columns:
            tmp[col] = tmp[col].astype(object)

        # 3) NaN/NaT/pandas.NA 등을 전부 None으로 치환
        tmp = tmp.where(pd.notna(tmp), None)

        # 4) dict로 변환 후 엄격한 JSON dump (NaN 허용 금지)
        records = tmp.to_dict(orient="records")
        with path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2, allow_nan=False)

    save_json(train_df, out_dir / "train/train.json")
    save_json(val_df, out_dir / "validation/val.json")
    save_json(test_df, out_dir / "test/test.json")

    print(f"저장 완료: {out_dir / 'train.json'}, {out_dir / 'val.json'}, {out_dir / 'test.json'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="CSV path")
    parser.add_argument("--out_dir", type=str, required=True, help="directory to save train/val/test.json")
    parser.add_argument("--train_size", type=int, default=2500)
    parser.add_argument("--val_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    split_dataset(
        csv_path=args.csv_path,
        out_dir=args.out_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()