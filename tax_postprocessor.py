# -*- coding: utf-8 -*-
"""Conservative post-processing for the TaxParser table output (Python >= 3.8).

No OCR/model/network dependency for the JSON workflow. pandas is imported only
by the existing-service adapter. Never fills an amount from a declared total.
Structural fixes are inferences: they are recorded separately from formatting.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import re
import unicodedata
from decimal import Decimal, InvalidOperation, localcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

VERSION = "1.2.1"
COLUMNS = ('原凭证号', '税种', '品目名称', '税款所属时期', '入（退）库日期', '实缴（退）金额')
FOUR_COLUMNS = (COLUMNS[1], COLUMNS[3], COLUMNS[4], COLUMNS[5])
AMOUNT = COLUMNS[-1]
VOUCHER = COLUMNS[0]
logger = logging.getLogger(__name__)

# A complete numeric token, not an arbitrary digit substring. Bad grouping such
# as 70,22.77 and unknown precision such as 123.456 are deliberately rejected.
_INTEGER = r'(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)'
_UNSIGNED = _INTEGER + r'(?:\.[0-9]{1,2})?'
_MONEY = re.compile(r'(?:\([¥￥]?\s*' + _UNSIGNED + r'\)|[+-]?[¥￥]?\s*' + _UNSIGNED + r')')
_SEP = re.compile(r'[\s;|]+')
_DATE = r'[0-9]{4}\s*(?:[.\-/]\s*[0-9]{1,2}\s*[.\-/]\s*[0-9]{1,2}|年\s*[0-9]{1,2}\s*月\s*[0-9]{1,2}\s*日?)'
_DATES = re.compile(_DATE)
_PERIODS = re.compile(_DATE + r'\s*[-~～—至]\s*' + _DATE)
_FULL_DATE = re.compile(r'\s*' + _DATE + r'\s*\Z')
_FULL_PERIOD = re.compile(r'\s*' + _DATE + r'\s*[-~～—至]\s*' + _DATE + r'\s*\Z')
_VOUCHER_TOKEN = re.compile(r'[A-Za-z0-9][A-Za-z0-9()（）\-/]*')


def _blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, Decimal) and value.is_nan():
        return True
    if type(value).__name__ in ('NAType', 'NaTType'):
        return True
    return isinstance(value, str) and not value.strip()


def _text(value: Any) -> str:
    return '' if _blank(value) else str(value).strip()


def _normalize_date_whitespace(column: str, value: Any) -> Any:
    """Only compact whitespace in a complete, recognizable date/period.

    Example: '2026. 07.15' => '2026.07.15'. Does NOT remove spaces
    from narrative fields or multiple unverified date tokens. This function
    preserves numeric date components and their original separator style.
    """
    if not isinstance(value, str) or not value.strip():
        return value
    candidate = unicodedata.normalize('NFKC', value)
    pattern = _FULL_PERIOD if column == COLUMNS[3] else _FULL_DATE if column == COLUMNS[4] else None
    if pattern is None or pattern.fullmatch(candidate) is None:
        return value
    return re.sub(r'\s+', '', candidate)


def normalize_column(name: Any) -> str:
    """Only whitespace and bracket variants; no fuzzy renaming/data loss."""
    compact = re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(name)))
    lookup = {re.sub(r'\s+', '', unicodedata.normalize('NFKC', c)): c for c in COLUMNS}
    return lookup.get(compact, str(name))


def _prepare_money(value: Any) -> str:
    s = unicodedata.normalize('NFKC', _text(value)).replace('−', '-')
    # Remove whitespace next to punctuation, not between arbitrary digits.
    # "70, 229.77 53,695.81" must remain TWO numbers.
    s = re.sub(r'(?<=[0-9])\s*([,.])\s*(?=[0-9])', r'\1', s)
    # A literal semicolon is a possible row separator, NOT automatically a
    # thousands comma. "29;247.65" is intrinsically ambiguous without context.
    return s


def parse_amounts(value: Any) -> Tuple[List[Decimal], Optional[str]]:
    """Parse the ENTIRE cell; return an error instead of guessing missing digits.

    Whitespace/semicolon/vertical bar can separate complete amounts. Spaces
    between digits are NOT assumed to be thousands separators. A merged row
    needs independent column evidence before these tokens may become new rows.
    """
    if _blank(value):
        return [], 'empty_amount'
    s = _prepare_money(value)
    if len(s) > 20000:
        return [], 'amount_cell_too_long'
    pos = 0
    values = []
    while pos < len(s):
        m = _MONEY.match(s, pos)
        if m is None:
            return [], 'unrecognized_amount_format'
        token = re.sub(r'[\s¥￥,]', '', m.group())
        if token.startswith('(') and token.endswith(')'):
            token = '-' + token[1:-1]
        # Guard pathological inputs; do not round arbitrary long values.
        if sum(ch.isdigit() for ch in token) > 30:
            return [], 'amount_too_large'
        try:
            number = Decimal(token)
            if not number.is_finite():
                return [], 'non_finite_amount'
        except InvalidOperation:
            return [], 'unrecognized_amount_format'
        values.append(number)
        pos = m.end()
        if pos == len(s):
            break
        sep = _SEP.match(s, pos)
        if sep is None or sep.end() == len(s):
            return [], 'unrecognized_amount_separator'
        pos = sep.end()
    return values, None


def format_amount(value: Decimal, thousands: bool = True) -> str:
    with localcontext() as ctx:
        ctx.prec = 40
        # Parsing already limited the number of decimal places: no rounding
        # of OCR values is performed here.
        return format(value, ',.2f' if thousands else '.2f')


def _change(report: Dict, code: str, rows: Sequence[int], before: Any,
            after: Any, reason: str, inferred: bool = False, **extra: Any) -> None:
    event = {
        'code': code, 'source_rows': list(rows), 'before': before, 'after': after,
        'reason': reason, 'inferred': inferred,
    }
    event.update(extra)
    report['changes'].append(event)


def _warning(report: Dict, code: str, message: str, **extra: Any) -> None:
    event = {'code': code, 'message': message}
    event.update(extra)
    report['warnings'].append(event)


def _finish(report: Dict) -> Dict:
    report['changed'] = bool(report.get('changes'))
    report['has_inferred_repairs'] = any(c.get('inferred') for c in report.get('changes', []))
    report['needs_review'] = bool(report.get('warnings')) or report['has_inferred_repairs']
    return report


def _full_pattern_parts(value: str, pattern: Any, count: int) -> Optional[List[str]]:
    matches = list(pattern.finditer(value))
    if len(matches) != count:
        return None
    pos = 0
    parts = []
    for match in matches:
        if value[pos:match.start()].strip():
            return None
        parts.append(match.group().strip())
        pos = match.end()
    return parts if not value[pos:].strip() else None


def _split_cell(column: str, value: Any, count: int) -> Optional[List[str]]:
    s = _text(value)
    if not s:
        return [''] * count
    if column == COLUMNS[3]:
        dates = _full_pattern_parts(s, _PERIODS, count)
        return [_normalize_date_whitespace(column, x) for x in dates] if dates is not None else None
    if column == COLUMNS[4]:
        dates = _full_pattern_parts(s, _DATES, count)
        return [_normalize_date_whitespace(column, x) for x in dates] if dates is not None else None
    parts = s.split()
    if len(parts) != count:
        return None
    if column == VOUCHER and not all(_VOUCHER_TOKEN.fullmatch(p) for p in parts):
        return None
    return parts


def _anchor(row: Dict) -> Optional[Tuple[str, ...]]:
    """A conservative local neighborhood: same tax, item, period and date."""
    if not _text(row.get('税种')) or not _text(row.get('税款所属时期')):
        return None
    return tuple(
        _text(_normalize_date_whitespace(k, row.get(k))) if k in (COLUMNS[3], COLUMNS[4])
        else _text(row.get(k))
        for k in COLUMNS[1:5]
    )


# A high-confidence repair must consume *observed* tokens, not fill in values
# based only on the previous row. A source with 2 tokens may supply at most one
# immediately following empty cell, a source with 3 tokens at most two, etc.
_MAX_SPILL_ROWS = 12
_TAX_TEXT = re.compile(r'[\u4e00-\u9fff]{1,30}(?:税|附加|费|基金)\Z')
# Whitelist of complete, commonly occurring tax item names.  Unknown item
# names are still supported when the OCR repeats the exact same full name;
# different unknown free-form fragments are not split without OCR coordinates.
_KNOWN_ITEMS = frozenset({
    '工资薪金所得', '劳务报酬所得', '稿酬所得', '特许权使用费所得',
    '利息股息红利所得', '股息红利所得', '财产租赁所得', '财产转让所得',
    '经营所得', '偶然所得', '其他所得', '综合所得',
    '工资薪金', '劳务报酬', '销售货物', '销售服务',
})


def _spill_parts(column: str, value: Any, thousands: bool) -> Optional[List[str]]:
    """Identify several complete values in one cell; never infer missing text.

    Column-specific parsing prevents blindly interpreting every space as a
    row boundary. Free-form text is only split when it contains repeated,
    exactly identical tokens; different tax types require complete tax names.
    """
    s = _text(value)
    if not s or len(s) > 20000:
        return None

    if column == AMOUNT:
        numbers, error = parse_amounts(s)
        if error or not 2 <= len(numbers) <= _MAX_SPILL_ROWS:
            return None
        # Bare integer fragments are inherently ambiguous (e.g. OCR-inserted
        # spaces inside one amount). Require explicit cents for each amount.
        if any(n.as_tuple().exponent != -2 for n in numbers):
            return None
        return [format_amount(n, thousands) for n in numbers]

    if column in (COLUMNS[3], COLUMNS[4]):
        pattern = _PERIODS if column == COLUMNS[3] else _DATES
        for n in range(2, _MAX_SPILL_ROWS + 1):
            parts = _full_pattern_parts(s, pattern, n)
            if parts is not None:
                return [_normalize_date_whitespace(column, p) for p in parts]
        return None

    if column not in (VOUCHER, COLUMNS[1], COLUMNS[2]):
        return None
    tokens = s.split()
    if not 2 <= len(tokens) <= _MAX_SPILL_ROWS:
        return None

    if column == VOUCHER:
        # Preserve v1.1's conservative identical-ID rule. Moving different
        # identifiers is not reliable without OCR cell coordinates.
        if len(set(tokens)) != 1 or not all(_VOUCHER_TOKEN.fullmatch(t) for t in tokens):
            return None
    elif column == COLUMNS[1]:
        # Repeated labels, or independently complete tax names separated by
        # whitespace, e.g. 个人所得税 增值税. Do not split 个人 所得税.
        if not all(_TAX_TEXT.fullmatch(t) for t in tokens):
            return None
    elif column == COLUMNS[2]:
        # 品目名称不是封闭词表：支持重复的完整名称，或所有片段均属于已知
        # 完整税目名称；不猜测任意自由文本的空格是否代表跨行边界。
        if not (all(len(t) >= 2 for t in tokens) and
                (len(set(tokens)) == 1 or all(t in _KNOWN_ITEMS for t in tokens))):
            return None
    return tokens


def _compare_spill_context(source: Dict, target: Dict,
                           spill_columns: Sequence[str]) -> List[str]:
    """Independent fields proving two adjacent records belong together.

    A difference in any *populated* non-spilled context column blocks the move.
    Amounts are never used as an equal-value anchor: they are independently
    checked for singularity and retained without numerical modification.
    """
    excluded = set(spill_columns)
    matched = []
    for col in COLUMNS[:-1]:
        if col in excluded:
            continue
        left = _text(_normalize_date_whitespace(col, source.get(col)))
        right = _text(_normalize_date_whitespace(col, target.get(col)))
        if left and right:
            if left != right:
                return []
            matched.append(col)
    return matched


def _repair_cross_row_spills(rows: List[Dict], origins: List[int],
                             report: Dict, thousands: bool = True) -> None:
    """Redistribute clearly concatenated cell values into adjacent empty rows.

    This handles each of the six standard columns, including multiple columns
    spilling into the same next row. A transfer must satisfy all safeguards:
      - 2..12 recognizable tokens in the source, matching count of empty cells;
      - existing adjacent rows, each with strong independent field agreement;
      - no modification to populated destination cells;
      - if amounts aren't spilled, one valid amount per participating row.
    There is no forward fill, heuristic copying, or total-based backsolving.
    """
    for i, source in enumerate(rows):
        if not isinstance(source, dict):
            continue
        detected = {}
        for col in COLUMNS:
            if col not in source:
                continue
            parts = _spill_parts(col, source[col], thousands)
            if parts is not None:
                detected[col] = parts

        # Keep the pre-existing warning about unequal/ambiguous voucher IDs.
        voucher_tokens = _text(source.get(VOUCHER)).split()
        if len(voucher_tokens) > 1 and VOUCHER not in detected:
            _warning(report, 'ambiguous_voucher_tokens',
                     '凭证号包含多个值，但无法可靠确定各值归属，未自动修改。',
                     source_row=origins[i], column=VOUCHER)
        # If another column *looks* as if it may contain a second value, but
        # the column-specific parser cannot establish two complete tokens,
        # do not silently repair just one of several potentially misplaced
        # columns. Record it for human review instead of guessing a split.
        ambiguous_columns = []
        if i + 1 < len(rows):
            adjacent = rows[i + 1]
            for col in COLUMNS:
                if col in detected or col not in source or not _blank(adjacent.get(col)):
                    continue
                raw = _text(source.get(col))
                possible_separator = len(raw.split()) > 1 or (col == AMOUNT and (';' in raw or '|' in raw))
                if not possible_separator:
                    continue
                same_context = _compare_spill_context(source, adjacent, list(detected) + [col])
                if (len(same_context) >= 2 and
                        any(c in same_context for c in (VOUCHER, COLUMNS[3], COLUMNS[4]))):
                    ambiguous_columns.append(col)
        if ambiguous_columns:
            _warning(report, 'ambiguous_possible_cross_row_spill',
                     '相邻空单元格对应的源单元格疑似有多个值，但无法可靠拆分；为避免部分错位修复，保留原值并建议核对原图。',
                     source_row=origins[i], columns=ambiguous_columns,
                     other_detected_columns=list(detected))
            continue
        if not detected:
            continue

        # Only participate in a simultaneous move when all detected columns
        # need the same number of destination rows. Disagreeing counts are
        # ambiguous, so do not perform partial moves.
        counts = {len(p) for p in detected.values()}
        if len(counts) != 1:
            _warning(report, 'conflicting_spill_counts',
                     '多个单元格疑似串行，但识别出的条目数量不一致，保留原值。',
                     source_row=origins[i], columns=list(detected))
            continue
        count = counts.pop()
        targets = rows[i + 1:i + count]
        if len(targets) != count - 1 or any(
            not all(_blank(target.get(col)) for target in targets) for col in detected
        ):
            _warning(report, 'unresolved_cross_row_spill',
                     '缺少数量对应的紧邻空单元格；没有移动或复制任何值。',
                     source_row=origins[i], columns=list(detected))
            continue

        # If source voucher is duplicated but next row already contains a
        # correct voucher, this column is not transferable and is not an anchor.
        # Other candidate columns must still stand on their own evidence.
        matched_by_row = [_compare_spill_context(source, target, list(detected))
                          for target in targets]
        voucher_only = tuple(detected) == (VOUCHER,)
        def enough_evidence(matched: List[str]) -> bool:
            # A unique structured identifier/date/period PLUS at least one
            # independent other field is required. Tax, item, date, amount, and
            # period spills demand >=3 matches; preserve v1.1 voucher behavior
            # with two independent matches (such as tax + period).
            threshold = 2 if voucher_only else 3
            return (len(matched) >= threshold and
                    any(c in matched for c in (VOUCHER, COLUMNS[3], COLUMNS[4])))
        if not all(enough_evidence(matched) for matched in matched_by_row):
            _warning(report, 'insufficient_spill_evidence',
                     '与相邻行的独立字段证据不足或存在冲突，未修改。',
                     source_row=origins[i], columns=list(detected),
                     matches=matched_by_row)
            continue

        # Parsed amounts prove these are separately recognized rows. If amount
        # itself is spilling, token count and precision were already verified.
        if AMOUNT not in detected:
            amount_ok = True
            for row in [source] + list(targets):
                amounts, error = parse_amounts(row.get(AMOUNT))
                if error or len(amounts) != 1:
                    amount_ok = False
                    break
            if not amount_ok:
                _warning(report, 'invalid_spill_amount_evidence',
                         '相邻行缺少独立的单个金额，未修改其他字段。',
                         source_row=origins[i], columns=list(detected))
                continue

        # Apply every column in the group atomically only after validation.
        for col, values in detected.items():
            affected = rows[i:i + count]
            before = [_text(r.get(col)) for r in affected]
            for j, part in enumerate(values):
                rows[i + j][col] = part
            code = ('redistribute_voucher_tokens' if col == VOUCHER
                    else 'redistribute_column_tokens')
            _change(report, code, origins[i:i + count], before, list(values),
                    '只移动已有的多个识别值至紧邻空单元格，且独立列匹配；无原图坐标，仍需人工核对。',
                    inferred=True, column=col,
                    output_rows=list(range(i + 1, i + count + 1)),
                    matched_context=matched_by_row)

        if AMOUNT in detected:
            # These warnings were emitted during first-pass amount handling,
            # but the multi-amount cell and target blanks have now been resolved.
            affected_origins = set(origins[i:i + count])
            report['warnings'] = [w for w in report['warnings']
                                  if not (w.get('source_row') in affected_origins and
                                          w.get('code') in ('empty_amount', 'ambiguous_merged_row'))]


# Keep the old helper name available for imports; repair_table uses the
# generalized function above to fix all supported columns in a single pass.
def _repair_voucher_spill(rows: List[Dict], origins: List[int], report: Dict) -> None:
    _repair_cross_row_spills(rows, origins, report)

def validate_total(rows: Sequence[Dict], declared_total: Any) -> Dict:
    """Result/page-level consistency check, not a certificate of OCR correctness."""
    source = declared_total
    if isinstance(source, (list, tuple)):
        # The existing API uses [Chinese amount, numeric amount].
        source = source[1] if len(source) > 1 else (source[0] if source else '')
    expected, total_error = parse_amounts(source)
    amounts = []
    unresolved = []
    for index, row in enumerate(rows, 1):
        nums, err = parse_amounts(row.get(AMOUNT))
        if err is not None or len(nums) != 1:
            unresolved.append(index)
        else:
            amounts.extend(nums)
    complete = not unresolved and bool(rows)
    with localcontext() as ctx:
        ctx.prec = 50
        partial_sum = sum(amounts, Decimal('0'))
        expected_value = expected[0] if total_error is None and len(expected) == 1 else None
        difference = partial_sum - expected_value if complete and expected_value is not None else None
    if not complete:
        status = 'unverifiable_rows'
    elif expected_value is None:
        status = 'unavailable_total'
    else:
        status = 'match' if difference == 0 else 'mismatch'
    return {
        'status': status, 'scope': 'result_or_page',
        'declared_total': format_amount(expected_value) if expected_value is not None else None,
        'row_sum': format_amount(partial_sum) if complete else None,
        'parsed_row_sum': format_amount(partial_sum), 'unresolved_rows': unresolved,
        'difference': format_amount(difference, False) if difference is not None else None,
        'note': '仅校验金额加和；不能证明行对应关系正确。多页文件的合计可能是整份文件合计。',
    }


def repair_table(rows: Sequence[Dict], declared_total: Any = None,
                 thousands: bool = True, check_total: bool = True) -> Tuple[List[Dict], Dict]:
    """Pure function: returns new rows plus a traceable report; never mutates input."""
    if not isinstance(rows, (list, tuple)) or any(not isinstance(r, dict) for r in rows):
        raise TypeError('table must be a list/tuple of dictionaries')
    report = {'version': VERSION, 'input_rows': len(rows), 'changes': [], 'warnings': []}
    output = []
    origins = []
    for source_index, original in enumerate(rows, 1):
        row = copy.deepcopy(original)
        if all(_blank(v) for v in row.values()):
            _change(report, 'remove_empty_row', [source_index], row, None, '仅删除所有字段均为空的行。')
            continue
        # Do not coerce cell types unrelated to the requested repair.
        normalized = {}
        collision = False
        for key, value in row.items():
            target = normalize_column(key)
            if target in normalized:
                collision = True
                break
            normalized[target] = value
        if collision:
            _warning(report, 'column_collision', '列名规范化后重名，整行保持原样。', source_row=source_index)
            output.append(row)
            origins.append(source_index)
            continue
        if list(normalized) != list(row):
            _change(report, 'normalize_column_names', [source_index], list(row), list(normalized),
                    '仅统一列名中的空白和全半角括号。')
        row = normalized
        # Normalize visual spacing in fully recognized date/period fields.
        # This also makes the local row-identity check insensitive to '2026. 07.15'.
        # All other fields and ambiguous date fragments remain unchanged.
        for date_column in (COLUMNS[3], COLUMNS[4]):
            if date_column in row:
                normalized_date = _normalize_date_whitespace(date_column, row[date_column])
                if normalized_date != row[date_column]:
                    _change(report, 'normalize_date_whitespace', [source_index],
                            row[date_column], normalized_date,
                            '仅清理完整日期/所属时期中的格式空格，不修改年月日数值。',
                            column=date_column)
                    row[date_column] = normalized_date
        if AMOUNT not in row:
            _warning(report, 'missing_amount_column', '未找到标准金额列，未猜测其他列的含义。', source_row=source_index)
            output.append(row)
            origins.append(source_index)
            continue
        nums, err = parse_amounts(row.get(AMOUNT))
        if err:
            _warning(report, err, '金额为空或格式不确定，保留原值，不根据合计反推。',
                     source_row=source_index, value=row.get(AMOUNT))
            output.append(row)
            origins.append(source_index)
            continue
        if len(nums) == 1:
            formatted = format_amount(nums[0], thousands)
            if row[AMOUNT] != formatted:
                _change(report, 'normalize_amount', [source_index], row[AMOUNT], formatted,
                        '规范千位逗号、小数点附近空格及两位小数；不改变数值。')
                row[AMOUNT] = formatted
            output.append(row)
            origins.append(source_index)
            continue
        count = len(nums)
        if any(number.as_tuple().exponent != -2 for number in nums):
            _warning(report, 'ambiguous_amount_tokens',
                     '多金额中存在没有明确两位小数的片段，可能是数字内部空格，未自动拆行。',
                     source_row=source_index, value=row.get(AMOUNT))
            output.append(row)
            origins.append(source_index)
            continue
        if count > 100:
            _warning(report, 'too_many_merged_rows', '单元格包含过多金额，未自动拆分。', source_row=source_index)
            output.append(row)
            origins.append(source_index)
            continue
        parts_by_column = {AMOUNT: [format_amount(n, thousands) for n in nums]}
        blocked = []
        structured_evidence = []
        nonempty_evidence = []
        for column, value in row.items():
            if column == AMOUNT:
                continue
            parts = _split_cell(column, value, count)
            if parts is None:
                blocked.append(column)
            else:
                parts_by_column[column] = parts
                if _text(value):
                    nonempty_evidence.append(column)
                    if column in (VOUCHER, COLUMNS[3], COLUMNS[4]):
                        structured_evidence.append(column)
        # At least TWO independently populated columns, including one structured
        # field (ID/date/period), must agree. No broadcasting of singleton text.
        if blocked or len(nonempty_evidence) < 2 or not structured_evidence:
            _warning(report, 'ambiguous_merged_row',
                     '多金额无法与其余列可靠逐一对应，整行保留；需要原图或 OCR 坐标。',
                     source_row=source_index, amount_count=count, blocked_columns=blocked)
            output.append(row)
            origins.append(source_index)
            continue
        split_rows = [{column: parts_by_column[column][j] for column in row} for j in range(count)]
        _change(report, 'split_merged_row', [source_index], copy.deepcopy(row), copy.deepcopy(split_rows),
                '金额个数与多列完整字段个数一致，按原始顺序拆分。无坐标证据，仍属于结构推断。',
                inferred=True, evidence_columns=nonempty_evidence)
        output.extend(split_rows)
        origins.extend([source_index] * count)
    _repair_cross_row_spills(output, origins, report, thousands=thousands)
    report['output_rows'] = len(output)
    report['output_source_rows'] = origins
    if check_total:
        report['total_check'] = validate_total(output, declared_total)
        status = report['total_check']['status']
        if status != 'match':
            _warning(report, 'total_' + status, '未通过金额合计校验（或无法校验），未修改明细来凑合计。')
    return output, _finish(report)


def repair_response(payload: Dict, thousands: bool = True,
                    include_report: bool = False) -> Tuple[Dict, Dict]:
    """Repair the existing mock-API JSON schema; keep errors/non-success untouched."""
    if not isinstance(payload, dict) or not isinstance(payload.get('data'), list):
        raise ValueError('Expected a response object with a data list')
    fixed = copy.deepcopy(payload)
    entries = []
    for file_index, document in enumerate(fixed['data']):
        if not isinstance(document, dict) or not isinstance(document.get('results'), list):
            continue
        for result_index, result in enumerate(document['results']):
            if not isinstance(result, dict):
                continue
            code = result.get('status', {}).get('code')
            entry = {'file_index': file_index, 'filename': document.get('filename'), 'result_index': result_index}
            if code != 200 or not isinstance(result.get('table'), list):
                entry.update({'skipped': True, 'reason': 'non_success_or_no_table', 'original_status': result.get('status')})
            else:
                table, report = repair_table(result['table'], result.get('content', {}).get('金额合计'), thousands)
                result['table'] = table
                entry.update(report)
                if include_report:
                    result['postprocess'] = copy.deepcopy(report)
            entries.append(entry)
    return fixed, {'version': VERSION, 'results': entries}


def _dataframe_records(dataframe: Any) -> Tuple[Optional[List[Dict]], Optional[List[str]], List[Dict]]:
    """Adapt numeric / first-row headers without throwing away unknown columns."""
    from difflib import SequenceMatcher
    notes = []
    if dataframe is None:
        return None, None, [{'code': 'no_dataframe', 'message': '只有 HTML、没有 DataFrame，保留原表。'}]
    columns = list(dataframe.columns)
    if not columns:
        return None, None, [{'code': 'no_columns', 'message': 'DataFrame 没有列，保留原表。'}]
    proper = not all(isinstance(c, int) or str(c).isdigit() for c in columns)
    start = 0
    if not proper:
        if dataframe.empty:
            return None, None, [{'code': 'empty_numeric_dataframe', 'message': '空表且无表头，保留原表。'}]
        candidates = [normalize_column(_text(v)) for v in dataframe.iloc[0].tolist()]
        score = max(SequenceMatcher(None, ''.join(candidates), ''.join(t)).ratio()
                    for t in (COLUMNS, FOUR_COLUMNS))
        if score >= 0.8:
            columns = candidates
            start = 1
        elif len(columns) in (4, 6):
            columns = list(FOUR_COLUMNS if len(columns) == 4 else COLUMNS)
            notes.append({'code': 'positional_header', 'message': '数字列名且首行不像表头，沿用原项目的四/六列位置模板；请核对列顺序。'})
        else:
            return None, None, [{'code': 'ambiguous_header', 'message': '无法确定列名，保留原始 DataFrame/HTML。'}]
    columns = [normalize_column(c) for c in columns]
    if len(set(columns)) != len(columns) or AMOUNT not in columns:
        return None, None, [{'code': 'ambiguous_columns', 'message': '列名重复或金额列无法识别，保留原表。'}]
    records = []
    for i in range(start, len(dataframe)):
        records.append({c: _text(dataframe.iloc[i, j]) for j, c in enumerate(columns)})
    return records, columns, notes


def postprocess_internal_results(results: Sequence[Dict], thousands: bool = True) -> List[Dict]:
    """Single integration point BEFORE frontend/API branches of app_test.py.

    Both the OCR path and the PDF text-extraction path end here. The API routes,
    model parameters, HTTP statuses and non-table fields remain unchanged.
    Reports stay internal by default; the installer can expose them via an env
    flag without changing the default API schema.
    """
    import pandas as pd
    processed = []
    for original in results:
        if original.get('status') != 'success' or original.get('_tax_postprocessed_version') == VERSION:
            processed.append(original)
            continue
        result = dict(original)
        report = {'version': VERSION, 'changes': [], 'warnings': [], 'tables': {}}
        table_results = original.get('table_results')
        if not isinstance(table_results, dict):
            processed.append(original)
            continue
        updated = dict(table_results)
        all_rows = []
        incomplete = False
        for table_type, table_data in table_results.items():
            if not isinstance(table_data, dict) or not table_data:
                continue  # preserve model placeholders
            try:
                records, columns, notes = _dataframe_records(table_data.get('dataframe'))
                if records is None:
                    incomplete = True
                    report['warnings'].extend(dict(n, table_type=table_type) for n in notes)
                    continue
                fixed, table_report = repair_table(records, thousands=thousands, check_total=False)
                table_report['warnings'].extend(notes)
                _finish(table_report)
                frame = pd.DataFrame(fixed, columns=columns)
                new_table = dict(table_data)
                new_table['dataframe'] = frame
                new_table['html'] = frame.to_html(index=False, border=0, escape=True)
                updated[table_type] = new_table
                all_rows.extend(fixed)
                report['tables'][table_type] = table_report
                report['changes'].extend(dict(c, table_type=table_type) for c in table_report['changes'])
                report['warnings'].extend(dict(w, table_type=table_type) for w in table_report['warnings'])
            except Exception as exc:
                # Preserve the original table and make failure visible. Do not
                # let a repair fail the whole OCR request.
                incomplete = True
                _warning(report, 'postprocess_exception', '该表后处理失败，已保留原始表格。',
                         table_type=table_type, error_type=type(exc).__name__)
        total = result.get('extracted_fields', {}).get('金额合计')
        report['total_check'] = validate_total(all_rows, total)
        if incomplete:
            report['total_check']['status'] = 'unverifiable_tables'
            report['total_check']['row_sum'] = None
            report['total_check']['difference'] = None
        if report['total_check']['status'] != 'match':
            _warning(report, 'total_' + report['total_check']['status'], '金额合计未匹配或不可验证；未改数值凑合计。')
        _finish(report)
        result['table_results'] = updated
        result['_postprocess_report'] = report
        result['_tax_postprocessed_version'] = VERSION
        # Summary only: do not add taxpayer IDs or amounts to logs here.
        log = logger.warning if report['needs_review'] else logger.info
        log('Tax table postprocess v%s: changes=%s inferred=%s warnings=%s total_check=%s',
            VERSION, len(report['changes']), report['has_inferred_repairs'], len(report['warnings']),
            report['total_check']['status'])
        processed.append(result)
    return processed


def main() -> None:
    parser = argparse.ArgumentParser(description='修复 TaxParser 响应中的金额格式、合并行和六列表格跨行串行。')
    parser.add_argument('input', type=Path, help='原始响应 JSON')
    parser.add_argument('-o', '--output', type=Path, required=True, help='修复后的 JSON，不能覆盖输入')
    parser.add_argument('--report', type=Path, help='审计报告 JSON，默认输出文件名加 .report.json')
    parser.add_argument('--no-thousands', action='store_true', help='输出 12345.67 而非 12,345.67')
    parser.add_argument('--include-report', action='store_true', help='额外在每个成功 result 中加入 postprocess 报告')
    args = parser.parse_args()
    report_path = args.report or args.output.with_suffix('.report.json')
    paths = [args.input.resolve(), args.output.resolve(), report_path.resolve()]
    if len(set(paths)) != 3:
        parser.error('输入、输出、报告必须是不同路径')
    try:
        payload = json.loads(args.input.read_text(encoding='utf-8-sig'))
        fixed, report = repair_response(payload, not args.no_thousands, args.include_report)
        for path, content in ((args.output, fixed), (report_path, report)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    except (OSError, ValueError, TypeError) as exc:
        parser.exit(1, '处理失败: %s\n' % exc)
    entries = report['results']
    print(json.dumps({
        'processed_results': sum(not e.get('skipped', False) for e in entries),
        'skipped_results': sum(e.get('skipped', False) for e in entries),
        'total_matches': sum(e.get('total_check', {}).get('status') == 'match' for e in entries),
        'needs_review': sum(e.get('needs_review', False) for e in entries),
        'output': str(args.output), 'report': str(report_path),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
