# -*- coding: utf-8 -*-
"""
税收完税证明抽取脚本（v2: added fill_person）：支持两类样式：
1) 表格式完税证明：有“原凭证号/税种/品目名称/税款所属时期/入库日期/金额”的表格型票据。
2) 文书式完税证明：无“原凭证号”，主体为四列表格：税种/税款所属时期/入库日期/金额。

依赖：
    pip install pymupdf pandas openpyxl

用法：
    python tax_certificate_extractor.py ./pdf目录 ./out.xlsx

在 Notebook 中：
    from tax_certificate_extractor import extract_pdf, extract_folder
    result = extract_pdf("文书式完税证明1.pdf")
    result["details_df"]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import pandas as pd


DATE_RE = re.compile(r"\d{4}[.\-/年]\d{1,2}[.\-/月]\d{1,2}日?")
AMOUNT_RE = re.compile(r"^[¥￥]?\s*-?\d[\d,]*(?:\.\d+)?\s*$")
VOUCHER_RE = re.compile(r"^\d{12,}$")
TAX_ID_RE = re.compile(r"\b[0-9A-Z]{15,20}\b")
CERT_NO_RE = re.compile(r"(?:No\.?|No●)\s*([0-9A-Za-z]+)|([0-9]{2}\(\d{4}\)\d{2}\s*证明\s*\d+)")

TABLE_HEADER_WORDS = {"原凭证号", "原 凭 证 号", "品目名称", "品 目 名 称"}
DOC_HEADER_WORDS = {"本凭证不作纳税人记账", "手写无效"}
DETAIL_COLS = ["原凭证号", "税种", "品目名称", "税款所属时期", "入（退）库日期", "实缴（退）金额"]
DOC_DETAIL_COLS = ["税种", "税款所属时期", "入（退）库日期", "实缴（退）金额"]


# ----------------------------- 基础清洗 -----------------------------

def read_pdf_text(pdf_path: str | Path) -> str:
    """读取 PDF 文本层。此脚本针对文本型 PDF；扫描件需先 OCR。"""
    parts = []
    with fitz.open(str(pdf_path)) as doc:
        for page in doc:
            parts.append(page.get_text("text"))
    return "\n".join(parts)


def normalize_line(s: str) -> str:
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":")
    s = re.sub(r"[ \t]+", " ", s.strip())
    return s


def normalize_text(text: str) -> str:
    text = text.replace("⼈", "人").replace("⺠", "民")
    text = text.replace("￥", "¥")
    return "\n".join(normalize_line(x) for x in text.splitlines() if normalize_line(x))


def get_lines(text: str) -> List[str]:
    return [normalize_line(x) for x in normalize_text(text).splitlines() if normalize_line(x)]


def compact(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def clean_value(s: str) -> str:
    s = normalize_line(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_amount(s: str) -> Optional[float]:
    if not s:
        return None
    s = s.replace("¥", "").replace(",", "").strip()
    try:
        return float(s)
    except Exception:
        return None


def normalize_date(s: str) -> str:
    """把 2025 年 1 月 17 日 / 2024.12.12 等尽量归一到 yyyy-mm-dd；失败则返回清洗文本。"""
    raw = clean_value(s)
    m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", raw)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", raw)
    if m:
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return raw


def normalize_period(s: str) -> str:
    raw = clean_value(s)
    # 2024.11.01-2024.11.30 / 2024-10-01 至 2024-12-31
    dates = re.findall(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", raw)
    if len(dates) >= 2:
        return f"{normalize_date(dates[0])} 至 {normalize_date(dates[1])}"
    return raw


def line_has_date(s: str) -> bool:
    return bool(DATE_RE.search(s))


def is_amount_line(s: str) -> bool:
    return bool(AMOUNT_RE.match(s.replace(" ", "")))


def is_label_line(s: str) -> bool:
    c = compact(s)
    labels = [
        "税务机关", "填发日期", "纳税人名称", "纳税人识别号", "税种", "税款所属时期",
        "入(退)库日期", "入（退）库日期", "实缴(退)金额", "实缴（退）金额", "原凭证号",
        "品目名称", "金额合计", "备注", "税收完税证明", "中华人民共和国", "妥善保管",
        "填票人", "盖章", "收据联", "交纳税人作完税证明"
    ]
    return any(x in c for x in labels)


# ----------------------------- 类型判断与通用字段 -----------------------------

def classify_certificate(text: str) -> str:
    c = compact(text)
    if any(compact(x) in c for x in TABLE_HEADER_WORDS):
        return "表格式完税证明"
    return "文书式完税证明"


def extract_cert_no(lines: List[str], text: str) -> str:
    for line in lines:
        m = CERT_NO_RE.search(line)
        if m:
            return clean_value(m.group(1) or m.group(2))
    m = CERT_NO_RE.search(text)
    return clean_value(m.group(1) or m.group(2)) if m else ""


def first_after_label(lines: List[str], label: str, stop_labels: Tuple[str, ...] = ()) -> str:
    """适合 label 独占一行或 label: value 的场景。"""
    lab_c = compact(label)
    stop_c = tuple(compact(x) for x in stop_labels)
    for i, line in enumerate(lines):
        c = compact(line)
        if lab_c in c:
            # label:value 在同一行
            if ":" in line:
                val = line.split(":", 1)[1].strip()
                if val and compact(val) not in stop_c:
                    return val
            # 向后收集直到下一个 label
            vals = []
            for nxt in lines[i + 1:]:
                nc = compact(nxt)
                if stop_c and any(x in nc for x in stop_c):
                    break
                if nc == lab_c:
                    continue
                if not nxt:
                    continue
                vals.append(nxt)
                # 默认取第一个；若要多行，需要外部函数
                break
            return clean_value("".join(vals))
    return ""


def collect_after_label(lines: List[str], label: str, stop_labels: Tuple[str, ...]) -> str:
    lab_c = compact(label)
    stop_c = tuple(compact(x) for x in stop_labels)
    vals = []
    taking = False
    for line in lines:
        c = compact(line)
        if not taking and lab_c in c:
            taking = True
            if ":" in line:
                after = line.split(":", 1)[1].strip()
                if after:
                    vals.append(after)
            continue
        if taking:
            if any(x in c for x in stop_c):
                break
            if not is_label_line(line) or vals:
                vals.append(line)
    return clean_value("".join(vals))


def extract_taxpayer_id(lines: List[str], text: str) -> str:
    # 优先在“纳税人识别号”附近找
    for i, line in enumerate(lines):
        if "纳税人识别号" in compact(line):
            window = " ".join(lines[i:i + 4])
            m = TAX_ID_RE.search(window)
            if m:
                return m.group(0)
    m = TAX_ID_RE.search(text)
    return m.group(0) if m else ""


def extract_total(lines: List[str]) -> Tuple[str, str, Optional[float]]:
    total_cn = ""
    total_raw = ""
    for i, line in enumerate(lines):
        if "金额合计" in compact(line):
            # 收集接下来若干行，直到备注/税务机关/填票人
            buf = []
            for nxt in lines[i:i + 8]:
                if nxt != line and any(k in compact(nxt) for k in ["备注", "税务机关", "填票人", "盖章"]):
                    break
                buf.append(nxt)
            joined = " ".join(buf)
            amounts = list(re.finditer(r"[¥￥]?\s*-?\d[\d,]*(?:\.\d+)?", joined))
            if amounts:
                # 最后一个阿拉伯金额是合计金额；中文大写只取它之前的文本，避免把“备注”等后续标签带进去。
                last = amounts[-1]
                total_raw = last.group(0).replace(" ", "")
                cn_part = joined[:last.start()]
            else:
                cn_part = joined
            # 中文大写金额一般在“(大写)”后。
            cn = re.sub(r"金额合计\s*\(?大写\)?", "", cn_part)
            cn = cn.replace("人民币", "人民币").strip(" :")
            total_cn = clean_value(cn)
            break
    return total_cn, total_raw, clean_amount(total_raw)




def extract_fill_person(lines: List[str]) -> str:
    """抽取“填票人”。兼容“填票人 电子税务局”、"填 票 人" 下一行给值等形式。"""
    stop_keywords = [
        "备注", "第", "妥善保管", "收据联", "交纳税人", "中华人民共和国",
        "税收完税证明", "本凭证", "手写无效", "金额合计", "税务机关", "盖章"
    ]
    label_pat = re.compile(r"填\s*票\s*人\s*[:：]?\s*(.*)")
    for i, line in enumerate(lines):
        m = label_pat.search(line)
        if not m:
            continue

        # 情况 1：值与“填票人”在同一行，例如：填票人 电子税务局
        same_line_value = clean_value(m.group(1))
        if same_line_value and not any(k in compact(same_line_value) for k in stop_keywords):
            return same_line_value

        # 情况 2：值在后续行，例如：填 票 人 / 电子税务局
        for nxt in lines[i + 1:i + 6]:
            nc = compact(nxt)
            if not nc:
                continue
            if any(k in nc for k in stop_keywords):
                break
            if is_label_line(nxt):
                continue
            return clean_value(nxt)
    return ""


# ----------------------------- 文书式解析 -----------------------------

def parse_doc_style(lines: List[str], text: str) -> Dict[str, Any]:
    cert_no = extract_cert_no(lines, text)
    tax_authority = collect_after_label(lines, "税务机关", ("填发日期",))
    issue_date = normalize_date(first_after_label(lines, "填发日期", ("纳税人名称",)))
    taxpayer_name = collect_after_label(lines, "纳税人名称", ("纳税人识别号",))
    taxpayer_id = extract_taxpayer_id(lines, text)
    total_cn, total_raw, total_num = extract_total(lines)
    fill_person = extract_fill_person(lines)

    # 明细区：从“实缴金额”后，到“金额合计”前。
    start = None
    for i, line in enumerate(lines):
        c = compact(line)
        if "实缴" in c and "金额" in c:
            start = i + 1
            break
    end = len(lines)
    for i, line in enumerate(lines):
        if start is not None and i > start and "金额合计" in compact(line):
            end = i
            break

    detail_lines = lines[start:end] if start is not None else []
    rows = []
    i = 0
    while i < len(detail_lines):
        # 跳过空/标签残留
        if is_label_line(detail_lines[i]) and "滞纳" not in detail_lines[i]:
            i += 1
            continue

        tax_parts = []
        # 税种可能跨行，例如“城市维护建设税(滞纳” + “金)”
        while i < len(detail_lines) and not line_has_date(detail_lines[i]):
            if is_amount_line(detail_lines[i]):
                break
            tax_parts.append(detail_lines[i])
            i += 1
        if not tax_parts or i >= len(detail_lines):
            i += 1
            continue

        period = detail_lines[i]
        i += 1
        if i >= len(detail_lines):
            break
        storage_date = detail_lines[i]
        i += 1
        if i >= len(detail_lines):
            break
        amount = detail_lines[i].replace(" ", "")
        i += 1

        rows.append({
            "税种": clean_value("".join(tax_parts)),
            "税款所属时期": normalize_period(period),
            "入（退）库日期": normalize_date(storage_date),
            "实缴（退）金额": amount,
        })

    return {
        "document_type": "文书式完税证明",
        "certificate_no": cert_no,
        "issue_date": issue_date,
        "tax_authority": tax_authority,
        "taxpayer_id": taxpayer_id,
        "taxpayer_name": taxpayer_name,
        "total_amount_cn": total_cn,
        "total_amount_raw": total_raw,
        "total_amount": total_num,
        "fill_person": fill_person,
        "details": rows,
    }


# ----------------------------- 表格式解析 -----------------------------

def extract_table_issue_date(lines: List[str]) -> str:
    # 支持 “填发日期：2024年12月11日” 或 后面多行 2025 年 / 1 月 / 17 日
    for i, line in enumerate(lines):
        if "填发日期" in compact(line):
            window = " ".join(lines[i:i + 5])
            m = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日", window)
            if m:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
            m = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", window)
            if m:
                return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    return ""


def parse_table_style(lines: List[str], text: str) -> Dict[str, Any]:
    cert_no = extract_cert_no(lines, text)
    issue_date = extract_table_issue_date(lines)
    taxpayer_id = extract_taxpayer_id(lines, text)
    taxpayer_name = collect_after_label(lines, "纳税人名称", ("原凭证号", "原 凭 证 号"))
    if not taxpayer_name:
        # 兜底：纳税人名称后一行
        taxpayer_name = first_after_label(lines, "纳税人名称", ("原凭证号", "原 凭 证 号"))
    tax_authority = collect_after_label(lines, "税务机关", ("纳税人识别号", "纳税人名称", "原凭证号"))
    total_cn, total_raw, total_num = extract_total(lines)
    fill_person = extract_fill_person(lines)

    # 明细区：从“实缴(退)金额”后到“金额合计”前
    start = None
    for i, line in enumerate(lines):
        c = compact(line)
        if "实缴" in c and "金额" in c:
            start = i + 1
            break
    end = len(lines)
    for i, line in enumerate(lines):
        if start is not None and i > start and "金额合计" in compact(line):
            end = i
            break
    detail_lines = lines[start:end] if start is not None else []

    rows = []
    i = 0
    while i < len(detail_lines):
        if not VOUCHER_RE.match(compact(detail_lines[i])):
            i += 1
            continue
        voucher = compact(detail_lines[i])
        i += 1
        if i >= len(detail_lines):
            break
        tax_type = detail_lines[i]
        i += 1

        item_parts: List[str] = []
        period_parts: List[str] = []
        # 品目名称读到遇见日期为止；若同一行里“品目+日期”粘连，拆开。
        while i < len(detail_lines):
            line = detail_lines[i]
            m = DATE_RE.search(line)
            if m:
                prefix = line[:m.start()].strip()
                suffix = line[m.start():].strip()
                if prefix:
                    item_parts.append(prefix)
                period_parts.append(suffix)
                i += 1
                break
            else:
                item_parts.append(line)
                i += 1
        # 税款所属时期可能分两行：2024-10-01 至 / 2024-12-31
        while i < len(detail_lines):
            line = detail_lines[i]
            # 入库日期通常是一条单独日期；只有当 period 已经有两个日期后才停止
            current_dates = re.findall(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}", " ".join(period_parts))
            if len(current_dates) >= 2:
                break
            if line_has_date(line):
                period_parts.append(line)
                i += 1
                continue
            break
        if i >= len(detail_lines):
            break
        storage_date = detail_lines[i]
        i += 1
        if i >= len(detail_lines):
            break
        amount = detail_lines[i].replace(" ", "")
        i += 1

        rows.append({
            "原凭证号": voucher,
            "税种": clean_value(tax_type),
            "品目名称": clean_value("".join(item_parts)),
            "税款所属时期": normalize_period(" ".join(period_parts)),
            "入（退）库日期": normalize_date(storage_date),
            "实缴（退）金额": amount,
            "实缴（退）金额_数值": clean_amount(amount),
        })

    return {
        "document_type": "表格式完税证明",
        "certificate_no": cert_no,
        "issue_date": issue_date,
        "tax_authority": tax_authority,
        "taxpayer_id": taxpayer_id,
        "taxpayer_name": taxpayer_name,
        "total_amount_cn": total_cn,
        "total_amount_raw": total_raw,
        "total_amount": total_num,
        "fill_person": fill_person,
        "details": rows,
    }


# ----------------------------- 对外接口 -----------------------------

def extract_pdf_by_pages(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """按页处理 PDF，每页返回独立的解析结果（含 page_index / total_pages）。"""
    pdf_path = Path(pdf_path)
    results = []
    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)
        for page_num, page in enumerate(doc, 1):
            raw_text = page.get_text("text") or ""
            if not raw_text.strip():
                continue
            text = normalize_text(raw_text)
            lines = get_lines(text)
            doc_type = classify_certificate(text)
            if doc_type == "表格式完税证明":
                result = parse_table_style(lines, text)
            else:
                result = parse_doc_style(lines, text)
            result["source_file"] = pdf_path.name
            result["details_df"] = pd.DataFrame(result["details"])
            result["page_index"] = page_num
            result["total_pages"] = total_pages
            results.append(result)
    return results


def extract_pdf(pdf_path: str | Path) -> Dict[str, Any]:
    pdf_path = Path(pdf_path)
    raw_text = read_pdf_text(pdf_path)
    text = normalize_text(raw_text)
    lines = get_lines(text)
    doc_type = classify_certificate(text)
    if doc_type == "表格式完税证明":
        result = parse_table_style(lines, text)
    else:
        result = parse_doc_style(lines, text)
    result["source_file"] = pdf_path.name
    result["details_df"] = pd.DataFrame(result["details"])
    return result


def flatten_result(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    meta_keys = [
        "source_file", "document_type", "certificate_no", "issue_date", "tax_authority",
        "taxpayer_id", "taxpayer_name", "total_amount_cn", "total_amount_raw", "total_amount", "fill_person"
    ]
    meta = {k: result.get(k, "") for k in meta_keys}
    rows = []
    for detail in result.get("details", []):
        rows.append({**meta, **detail})
    if not rows:
        rows.append(meta)
    return rows


def extract_folder(input_path: str | Path, out_excel: str | Path | None = None) -> Dict[str, Any]:
    input_path = Path(input_path)
    pdfs = [input_path] if input_path.is_file() else sorted(input_path.glob("*.pdf"))
    results = []
    flat_rows = []
    for pdf in pdfs:
        res = extract_pdf(pdf)
        results.append(res)
        flat_rows.extend(flatten_result(res))

    summary_df = pd.DataFrame(flat_rows)

    if out_excel:
        out_excel = Path(out_excel)
        with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="all_details", index=False)
            meta_rows = []
            for r in results:
                meta_rows.append({k: v for k, v in r.items() if k not in {"details", "details_df"}})
            pd.DataFrame(meta_rows).to_excel(writer, sheet_name="metadata", index=False)
            for idx, r in enumerate(results, start=1):
                name = re.sub(r"[\\/*?:\[\]]", "_", Path(r["source_file"]).stem)[:20]
                sheet = f"{idx}_{name}"[:31]
                r["details_df"].to_excel(writer, sheet_name=sheet, index=False)
    return {"results": results, "summary_df": summary_df}


def result_to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in result.items() if k != "details_df"}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python tax_certificate_extractor.py <pdf文件或目录> [out.xlsx]")
        sys.exit(1)
    inp = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) >= 3 else Path("tax_certificate_extract.xlsx")
    pack = extract_folder(inp, out)
    print(pack["summary_df"])
    print(f"\n已导出：{out}")
