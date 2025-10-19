#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import pandas as pd

# ===== OpenAI (>=1.0) =====
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

# ------------------------
# Safe readers
# ------------------------
def safe_read_excel(path: str) -> pd.DataFrame:
    """엑셀을 안전하게 읽기 (.xlsx=openpyxl, .xls=xlrd)"""
    ext = Path(path).suffix.lower()
    if ext == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    elif ext == ".xls":
        return pd.read_excel(path, engine="xlrd")
    # 그 외 확장자라도 엑셀일 수 있으니 openpyxl 시도
    return pd.read_excel(path, engine="openpyxl")

def safe_read_columns(path: str) -> pd.DataFrame:
    """
    컬럼 메타 로더: CSV 또는 엑셀 자동 판별
    기대 컬럼: table_name (필수), column_name (선택), alt_name (선택)
    """
    ext = Path(path).suffix.lower()
    if ext == ".csv":
        df = pd.read_csv(path, dtype=str).fillna("")
    else:
        df = safe_read_excel(path)
        # 숫자/NaN 방지
        for c in df.columns:
            df[c] = df[c].astype(str)
        df = df.fillna("")
    return df

# =========================================
# 1) 로딩 & 정제: 테이블 메타 (첫 번째 엑셀)
# =========================================
NEEDED_TABLE_COLS = ["테이블영문명", "테이블국문명", "설명", "키워드"]

def _clean_description(desc: object) -> str:
    """여러 어절(공백 1개 이상)이면 유지, 아니면 빈 문자열"""
    if isinstance(desc, str):
        tokens = desc.strip().split()
        if len(tokens) > 1:
            return desc.strip()
    return ""

def load_tables_df(tables_xlsx: str) -> pd.DataFrame:
    df = pd.read_excel(tables_xlsx, engine="openpyxl")
    missing = [c for c in NEEDED_TABLE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[tables] 필요한 컬럼이 없습니다: {missing}")
    df = df[NEEDED_TABLE_COLS].copy()
    df["테이블영문명"] = df["테이블영문명"].astype(str).str.strip()
    df["테이블국문명"] = df["테이블국문명"].astype(str).str.strip()
    df["설명"] = df["설명"].apply(_clean_description)
    # 키워드: NaN -> "" + strip
    df["키워드"] = df["키워드"].astype(str).fillna("").str.strip()
    return df


# =========================================
# 2) 로딩 & 그룹화: 컬럼 메타 (두 번째 엑셀)
#    기대 컬럼: table_name, column_name(선택), alt_name
# =========================================
def _normalize_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    columns 엑셀 표준화:
      - table_name: 필수
      - alt_name:   선택(없으면 빈 문자열)
      - column_name: 선택(없으면 빈 문자열)
    """
    cols = df.columns
    if "table_name" not in cols:
        # 한국어 컬럼명 대응을 원한다면 아래 주석 해제하고 이름 맞춰도 됨.
        # if "테이블영문명" in cols: df = df.rename(columns={"테이블영문명": "table_name"})
        raise ValueError("[columns] 'table_name' 컬럼이 필요합니다.")

    # column_name/alt_name 없으면 생성
    if "column_name" not in cols:
        df["column_name"] = ""
    if "alt_name" not in cols:
        df["alt_name"] = ""

    out = df[["table_name", "column_name", "alt_name"]].copy()
    out["table_name"] = out["table_name"].astype(str).str.strip()
    out["column_name"] = out["column_name"].astype(str).fillna("").str.strip()
    out["alt_name"] = out["alt_name"].astype(str).fillna("").str.strip()

    # table_name 비어있는 행 제거
    out = out[out["table_name"] != ""]
    return out

def build_column_lists(columns_xlsx: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    반환:
      - df_cols:  table_name -> [column_name] 리스트
      - df_alts:  table_name -> [alt_name]    리스트
    """
    
    raw = safe_read_columns(columns_path)
    df = _normalize_columns_df(raw)

    # 원 컬럼명 리스트 (빈 문자열은 제외)
    df_cols = (
        df[df["column_name"] != ""]
        .groupby("table_name")["column_name"]
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index()
        .rename(columns={"column_name": "columns"})
    )

    # 대체명 리스트 (빈 문자열은 제외)
    df_alts = (
        df[df["alt_name"] != ""]
        .groupby("table_name")["alt_name"]
        .apply(lambda s: sorted(set(s.tolist())))
        .reset_index()
        .rename(columns={"alt_name": "alt_names"})
    )

    return df_cols, df_alts


# =========================================
# 3) 조인: 테이블 메타 + 컬럼/대체명 리스트
# =========================================
def join_table_and_columns(
    tables_df: pd.DataFrame, df_cols: pd.DataFrame, df_alts: pd.DataFrame
) -> pd.DataFrame:
    merged = tables_df.merge(
        df_cols, left_on="테이블영문명", right_on="table_name", how="left"
    ).drop(columns=["table_name"], errors="ignore")

    merged = merged.merge(
        df_alts, left_on="테이블영문명", right_on="table_name", how="left"
    ).drop(columns=["table_name"], errors="ignore")

    # 리스트 필드 기본값
    if "columns" not in merged.columns:
        merged["columns"] = [[] for _ in range(len(merged))]
    else:
        merged["columns"] = merged["columns"].apply(lambda v: v if isinstance(v, list) else [])

    if "alt_names" not in merged.columns:
        merged["alt_names"] = [[] for _ in range(len(merged))]
    else:
        merged["alt_names"] = merged["alt_names"].apply(lambda v: v if isinstance(v, list) else [])

    return merged


# =========================================
# 4) GPT 호출 유틸 (배치/청크)
# =========================================
def approx_tokens_from_text(text: str) -> int:
    """대략 1토큰 ~= 4문자 가정"""
    return max(1, math.ceil(len(text) / 4))

def build_table_prompt_item(row: pd.Series) -> Dict[str, Any]:
    """한 테이블에 대해 프롬프트에 넣을 페이로드 생성"""
    return {
        "table_name": row["테이블영문명"],
        "table_name_ko": row["테이블국문명"],
        "description": row.get("설명", ""),
        "keywords": row.get("키워드", ""),
        "columns": row.get("columns", []),
        "alt_names": row.get("alt_names", []),
    }

def format_prompt_payload(items: List[Dict[str, Any]], user_instruction: str) -> str:
    """
    모델에 전달할 단일 문자열 프롬프트 구성 (JSON 블록 포함).
    user_instruction: 사용자가 원하는 산출물 스펙(예: 카테고리/추천 키워드/품질체크 규칙 등)
    """
    payload = {
        "instruction": user_instruction,
        "tables": items,
        "output_schema": {
            # 원하는 출력 스키마 예시(자유롭게 바꾸세요)
            "table_name": "string",
            "summary": "string",
            "suggested_keywords": ["string"],
            "column_groups": [{"group": "string", "columns": ["string"]}],
        },
        "format": "Return a strict JSON array aligned to output_schema. No extra commentary.",
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)

def split_batches(
    df: pd.DataFrame,
    user_instruction: str,
    max_tokens_per_batch: int = 8000,
    max_items_per_batch: int = 20,
) -> List[str]:
    """토큰/아이템 한도를 고려해 여러 프롬프트 배치로 분할"""
    batches = []
    cur_items: List[Dict[str, Any]] = []
    cur_tokens = 0

    def flush():
        nonlocal cur_items, cur_tokens
        if not cur_items:
            return
        prompt = format_prompt_payload(cur_items, user_instruction)
        batches.append(prompt)
        cur_items = []
        cur_tokens = 0

    # 대략적인 헤더 토큰 (instruction/스키마 고정분)
    header_tokens = approx_tokens_from_text(
        format_prompt_payload([], user_instruction)
    )

    for _, row in df.iterrows():
        item = build_table_prompt_item(row)
        item_text = json.dumps(item, ensure_ascii=False)
        item_tokens = approx_tokens_from_text(item_text)

        if (
            cur_items
            and (
                header_tokens + cur_tokens + item_tokens > max_tokens_per_batch
                or len(cur_items) >= max_items_per_batch
            )
        ):
            flush()

        cur_items.append(item)
        cur_tokens += item_tokens

    flush()
    return batches


def call_gpt(
    prompt: str,
    model: str = "gpt-4.1-mini",
    temperature: float = 0.2,
    max_output_tokens: int = 2000,
    retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """
    OpenAI Responses API 우선 사용. 실패 시 간단 재시도.
    반환: 모델의 '텍스트' 출력 (JSON 문자열 기대)
    """
    if not _HAS_OPENAI:
        raise RuntimeError("openai 모듈을 찾을 수 없습니다. `pip install openai` 후 재시도하세요.")

    client = OpenAI()

    last_err = None
    for _ in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                # JSON 강제는 모델/버전에 따라 변동될 수 있어 프롬프트로 엄격히 요구
            )
            # SDK는 output_text를 제공
            return resp.output_text
        except Exception as e:
            last_err = e
            time.sleep(retry_delay)
    raise RuntimeError(f"GPT 호출 실패: {last_err}")


def run_gpt_on_batches(
    batches: List[str],
    model: str,
    temperature: float = 0.2,
    max_output_tokens: int = 2000,
    sleep_between: float = 0.5,
) -> List[List[Dict[str, Any]]]:
    """
    각 배치를 GPT에 보내고, JSON 배열을 파싱하여 리스트로 반환.
    """
    results: List[List[Dict[str, Any]]] = []
    for i, prompt in enumerate(batches, 1):
        print(f"[GPT] 배치 {i}/{len(batches)} 호출 중...")
        text = call_gpt(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("JSON 루트가 배열이 아닙니다.")
            results.append(data)
        except Exception as e:
            # 모델이 포맷을 깨면 보정 시도(마크다운 코드펜스 제거 등)
            fixed = text.strip()
            if fixed.startswith("```"):
                fixed = fixed.strip("`")
                # 언어 힌트 제거
                if fixed.startswith("json"):
                    fixed = fixed[4:]
            try:
                data = json.loads(fixed)
                if not isinstance(data, list):
                    raise ValueError("JSON 루트가 배열이 아닙니다.")
                results.append(data)
            except Exception:
                # 디버깅을 위해 원문 남김
                raise ValueError(f"배치 {i} JSON 파싱 실패\n---\n{text}\n---\n에러: {e}")
        time.sleep(sleep_between)
    return results


# =========================================
# 5) 메인: 파이프라인
# =========================================
def main(
    tables_xlsx: str,
    columns_xlsx: str,
    out_dir: str = "out",
    model: str = "gpt-4.1-mini",
    max_tokens_per_batch: int = 8000,
    max_items_per_batch: int = 20,
    user_instruction: Optional[str] = None,
):
    """
    파이프라인:
      1) 테이블 메타 로드/정제
      2) 컬럼 메타 로드/그룹화 (두 DF: columns, alt_names)
      3) 조인 → 단일 DF
      4) 배치 분할 후 GPT 호출
      5) JSONL/CSV/XLSX 저장
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tables_df = load_tables_df(tables_xlsx)
    df_cols, df_alts = build_column_lists(columns_xlsx)
    merged = join_table_and_columns(tables_df, df_cols, df_alts)

    print(merged.head())
    return
    # 기본 지시문(원하면 바꾸세요)
    if not user_instruction:
        user_instruction = (
            "각 테이블의 목적을 한 문장으로 요약(summary)하고, "
            "columns/alt_names를 참고해 논리적 column_groups를 제안하세요. "
            "또한 검색을 위한 suggested_keywords 5~10개를 제시하세요. "
            "응답은 output_schema에 맞춘 JSON 배열로만 반환하세요."
        )

    batches = split_batches(
        merged,
        user_instruction=user_instruction,
        max_tokens_per_batch=max_tokens_per_batch,
        max_items_per_batch=max_items_per_batch,
    )

    print(f"[INFO] 총 {len(merged)}개 테이블, 배치 {len(batches)}개로 분할.")

    gpt_results = run_gpt_on_batches(
        batches=batches,
        model=model,
        temperature=0.2,
        max_output_tokens=2000,
    )

    # 합치기
    flat: List[Dict[str, Any]] = [item for batch in gpt_results for item in batch]

    # 결과 저장
    jsonl_path = Path(out_dir) / "gpt_results.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for obj in flat:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # CSV/XLSX 변환(스키마에 따라 평탄화)
    df_out = pd.json_normalize(flat)
    csv_path = Path(out_dir) / "gpt_results.csv"
    xlsx_path = Path(out_dir) / "gpt_results.xlsx"
    df_out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as w:
        df_out.to_excel(w, index=False, sheet_name="gpt_results")

    # 중간 결과(조인 데이터)도 저장해두면 디버깅 용이
    merged_path = Path(out_dir) / "merged_input.xlsx"
    with pd.ExcelWriter(merged_path, engine="xlsxwriter") as w:
        merged.copy().assign(
            columns_str=lambda d: d["columns"].apply(lambda xs: ", ".join(xs)),
            alt_names_str=lambda d: d["alt_names"].apply(lambda xs: ", ".join(xs)),
        ).to_excel(w, index=False, sheet_name="merged_input")

    print(f"[DONE] JSONL: {jsonl_path}")
    print(f"[DONE] CSV:   {csv_path}")
    print(f"[DONE] XLSX:  {xlsx_path}")
    print(f"[DONE] MERGED_INPUT: {merged_path}")


"""
Usage:
export OPENAI_API_KEY=sk-xxxx
python make_new_data.py \
  --tables "[4641830]20251019 IT-META(table).xlsx" \
  --columns "columns.xlsx" \
  --out out \
  --model gpt-4.1-mini \
  --max_tokens_per_batch 8000 \
  --max_items_per_batch 20
"""


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Tables+Columns → GPT 파이프라인")
    # parser.add_argument("--tables", required=True, help="첫 번째 엑셀 (테이블 메타)")
    # parser.add_argument("--columns", required=True, help="두 번째 엑셀 (컬럼 메타)")
    # parser.add_argument("--out", default="out", help="출력 디렉토리")
    # parser.add_argument("--model", default="gpt-4.1-mini", help="OpenAI 모델명")
    # parser.add_argument("--max_tokens_per_batch", type=int, default=8000)
    # parser.add_argument("--max_items_per_batch", type=int, default=20)
    # parser.add_argument("--instruction", default=None, help="사용자 지시문(없으면 기본값)")
    # args = parser.parse_args()

    args = {
        "tables":"./data/1019_IT-META(table).xlsx",
        "columns":"./data/1019_IT-META(columns).csv",
        "out": "./output"
    }
    ## 키 체크
    #if "OPENAI_API_KEY" not in os.environ:
    #    print("[WARN] OPENAI_API_KEY가 설정되지 않았습니다. `export OPENAI_API_KEY=...`")
    main(
        tables_xlsx=args.get('tables'),
        columns_xlsx=args.get('columns'),
        out_dir=args.get('out'),
        model=args.get('model'),
        max_tokens_per_batch=args.get('max_tokens_per_batch'),
        max_items_per_batch=args.get('max_items_per_batch'),
        user_instruction=args.get('instructions'),
    )

