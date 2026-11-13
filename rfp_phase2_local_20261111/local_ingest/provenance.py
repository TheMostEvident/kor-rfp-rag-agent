"""새로 파싱한 원문에서 평가 근거를 복구하고 출처 위치를 검증합니다.

이 모듈은 이전 텍스트 스냅샷을 읽지 않습니다. ``documents``는 원본 바이트를
새로 처리한 결과여야 하며, ``sha256``은 로더가 그 원본에서 계산해야 합니다.
문자열 복구 검증은 정답의 의미 정확도나 RAG의 답변 성능 평가와 다릅니다.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any


def _valid_hash(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        return None
    return value.lower()


def _occurrences(text: str, query: str):
    """겹치는 일치도 보존합니다. 여러 위치를 임의의 첫 위치로 확정하지 않습니다."""
    start = 0
    while query and start <= len(text):
        position = text.find(query, start)
        if position < 0:
            return
        yield position, position + len(query)
        start = position + 1


def _without_whitespace(text: str):
    # 공백 외의 문장부호·숫자·한글 코드 포인트는 바꾸지 않습니다.
    positions = [i for i, character in enumerate(text) if not character.isspace()]
    return "".join(text[i] for i in positions), positions


def _source_views(document: dict):
    views = []
    seen = set()
    for collection, reference in (("blocks", False), ("reference_blocks", True)):
        for block in document.get(collection, []) or []:
            if not isinstance(block, dict):
                continue
            text, locator = block.get("text"), block.get("locator")
            if not isinstance(text, str) or not isinstance(locator, str) or not text:
                continue
            # 같은 실제 위치의 같은 텍스트가 호환용 view에 다시 실린 경우입니다.
            # 서로 다른 locator에 같은 텍스트가 있는 경우는 제거하지 않습니다.
            key = (locator, text)
            if key in seen:
                continue
            seen.add(key)
            views.append({"block": block, "reference_view": reference})
    return views


def _find_matches(views, query, normalized=False, maximum=20):
    matches = []
    count = 0
    if normalized:
        query, _ = _without_whitespace(query)
    if not query:
        return matches, count
    for view in views:
        block = view["block"]
        text = block["text"]
        if normalized:
            if "normalized_text" not in view:
                view["normalized_text"], view["character_map"] = _without_whitespace(text)
            search_text = view["normalized_text"]
        else:
            search_text = text
        for start, end in _occurrences(search_text, query):
            count += 1
            if len(matches) >= maximum:
                continue
            if normalized:
                positions = view["character_map"]
                raw_start, raw_end = positions[start], positions[end - 1] + 1
            else:
                raw_start, raw_end = start, end
            matches.append({
                "block_id": block.get("block_id"),
                "locator": block["locator"],
                "char_start": raw_start,
                "char_end": raw_end,
                "matched_text": text[raw_start:raw_end],
                "reference_view": view["reference_view"],
                "block_kind": block.get("kind"),
                "source_method": block.get("source_method", "raw_parser_reference_view" if view["reference_view"] else "raw_parser"),
            })
    return matches, count


def _evidence_items(case):
    for fact in case.get("required_facts", []) or []:
        for index, evidence in enumerate(fact.get("evidence", []) or [], 1):
            yield fact.get("fact_id"), "fact", index, evidence
    for index, evidence in enumerate(case.get("case_basis_evidence", []) or [], 1):
        yield None, "case_basis", index, evidence


def _document_index(documents):
    index = {}
    for document in documents:
        if not isinstance(document, dict) or not document.get("doc_id"):
            raise ValueError("모든 파싱 결과에 doc_id가 필요합니다.")
        doc_id = document["doc_id"]
        if doc_id in index:
            raise ValueError(f"중복된 doc_id입니다: {doc_id}")
        index[doc_id] = document
    return index


def verify_evidence(documents: list[dict], cases: list[dict], *,
                    allow_whitespace_normalization: bool = True,
                    max_candidates: int = 20) -> dict:
    """모든 gold 인용 연결을 새 원문 결과에서 독립적으로 검증합니다.

    필수 입력은 doc_id·원본 sha256·blocks이며 호환용 reference_blocks를 사용할 수
    있습니다. 같은 내용의 별칭 문서에 blocks가 없다면 명시적 canonical_doc_id가
    가리키는 결과를 사용하되 양쪽 원본 해시가 같아야 합니다. 다른 문서의 텍스트를
    임의 검색하거나 같은 문구라는 이유로 해시가 다른 파일을 대신 사용하지 않습니다.

    우선순위는 원래 locator의 exact → 원래 locator의 공백 정규화 → 같은 문서의
    다른 위치 exact → 같은 문서의 다른 위치 공백 정규화입니다. 정규화는 Unicode
    whitespace만 제거하며, 반환 offset은 정규화 전 원문 문자열의 Python 문자
    인덱스(0부터, char_end 미포함)입니다. 정규화 일치를 exact로 집계하지 않습니다.

    여러 위치에 있으면 모든 후보 수를 기록하고 최대 max_candidates개를 반환합니다.
    단일 위치를 임의 선택하지 않습니다. exact_matches는 문자열 발견 수이며,
    위치까지 하나로 확정된 수는 resolved_matches와 location_resolved로 확인합니다.

    unmatched에는 해시 불일치·문서 없음·잘못된 근거도 포함되므로
    total == exact_matches + normalized_matches + unmatched입니다.
    근거가 없는 확인 질문/합성 실패 사례는 분모에 더하지 않습니다.
    """
    if not isinstance(max_candidates, int) or max_candidates < 1:
        raise ValueError("max_candidates는 1 이상의 정수여야 합니다.")
    index = _document_index(documents)
    views_by_id = {}
    records = []
    case_rows = []

    for case in cases:
        case_id = case.get("case_id")
        start_of_case = len(records)
        for fact_id, evidence_kind, evidence_index, evidence in _evidence_items(case):
            evidence = evidence if isinstance(evidence, dict) else {}
            doc_id = evidence.get("doc_id")
            quote = evidence.get("quote")
            expected_hash = _valid_hash(evidence.get("sha256"))
            gold_locator = evidence.get("locator")
            record = {
                "case_id": case_id,
                "fact_id": fact_id,
                "evidence_kind": evidence_kind,
                "evidence_index": evidence_index,
                "doc_id": doc_id,
                "gold_sha256": evidence.get("sha256"),
                "gold_block_id": evidence.get("block_id"),
                "gold_locator": gold_locator,
                "quote": quote,
                "status": "unmatched",
                "match_mode": "not_found",
                "source_doc_id": None,
                "file_name": None,
                "source_sha256": None,
                "current_locator": None,
                "block_id": None,
                "char_start": None,
                "char_end": None,
                "raw_exact": False,
                "reference_view": False,
                "location_resolved": False,
                "locator_status": "not_resolved",
                "candidate_count": 0,
                "candidates_truncated": False,
                "candidates": [],
                "provenance": None,
            }
            records.append(record)
            if not doc_id or not expected_hash or not isinstance(quote, str) or not quote.strip():
                record["match_mode"] = "invalid_evidence"
                record["reason"] = "doc_id, 유효한 원본 SHA256, 비어 있지 않은 quote가 필요합니다."
                continue
            document = index.get(doc_id)
            if document is None:
                record["match_mode"] = "missing_document"
                record["reason"] = "요청한 doc_id의 새 파싱 결과가 없습니다."
                continue
            record["file_name"] = document.get("file_name", document.get("name", doc_id))
            actual_hash = _valid_hash(document.get("sha256"))
            record["source_sha256"] = document.get("sha256")
            if actual_hash != expected_hash:
                record["match_mode"] = "hash_mismatch"
                record["reason"] = "평가 근거와 새로 읽은 원본의 SHA256이 다릅니다."
                continue
            source_document = document
            source_doc_id = doc_id
            # 명시적으로 연결된 동일 내용 별칭만 canonical 결과를 공유할 수 있습니다.
            if not document.get("blocks") and not document.get("reference_blocks"):
                canonical_id = document.get("canonical_doc_id", document.get("canonical_document_id"))
                if canonical_id and canonical_id != doc_id:
                    canonical = index.get(canonical_id)
                    if canonical is None or _valid_hash(canonical.get("sha256")) != actual_hash:
                        record["match_mode"] = "alias_source_mismatch"
                        record["reason"] = "별칭이 가리키는 원본 결과가 없거나 SHA256이 다릅니다."
                        continue
                    source_document, source_doc_id = canonical, canonical_id
            record["source_doc_id"] = source_doc_id
            if source_doc_id not in views_by_id:
                views_by_id[source_doc_id] = _source_views(source_document)
            views = views_by_id[source_doc_id]
            preferred = [v for v in views if v["block"]["locator"] == gold_locator] if gold_locator else []
            strategy = [(preferred, False, True)]
            if allow_whitespace_normalization:
                strategy.append((preferred, True, True))
            strategy.append((views, False, False))
            if allow_whitespace_normalization:
                strategy.append((views, True, False))
            selected = None
            for search_views, normalized, at_original_locator in strategy:
                if not search_views:
                    continue
                matches, count = _find_matches(search_views, quote, normalized, max_candidates)
                if count:
                    selected = matches, count, normalized, at_original_locator
                    break
            if selected is None:
                record["reason"] = "해당 문서의 새 파싱 본문/명시적 호환 view에서 인용을 찾지 못했습니다."
                continue
            matches, count, normalized, at_original_locator = selected
            record.update({
                "status": "matched" if count == 1 else "ambiguous",
                "match_mode": "normalized_whitespace" if normalized else "exact",
                "raw_exact": not normalized,
                "normalization": "unicode_whitespace_removed" if normalized else None,
                "candidate_count": count,
                "candidates": matches,
                "candidates_truncated": count > len(matches),
                "locator_status": "original" if at_original_locator else "relocated" if gold_locator else "gold_locator_missing",
                "location_resolved": count == 1,
            })
            if count == 1:
                match = matches[0]
                record.update({
                    "current_locator": match["locator"],
                    "block_id": match["block_id"],
                    "char_start": match["char_start"],
                    "char_end": match["char_end"],
                    "matched_text": match["matched_text"],
                    "reference_view": match["reference_view"],
                    "block_kind": match["block_kind"],
                    "source_method": match["source_method"],
                    "provenance": f"{record['file_name']} · {match['locator']} · chars[{match['char_start']}:{match['char_end']}]",
                })
            else:
                record["reference_view"] = count == len(matches) and all(m["reference_view"] for m in matches)
                record["reason"] = "인용 문자열은 복구했지만 가능한 원문 위치가 여러 개이므로 단일 위치로 확정하지 않습니다."
                record["provenance"] = f"{record['file_name']} · 원문 위치 후보 {count}개"

        case_records = records[start_of_case:]
        case_rows.append({
            "case_id": case_id,
            "evidence_links": len(case_records),
            "matched_links": sum(r["match_mode"] in {"exact", "normalized_whitespace"} for r in case_records),
            "resolved_links": sum(r["location_resolved"] for r in case_records),
            "has_source_evidence": bool(case_records),
        })

    modes = Counter(record["match_mode"] for record in records)
    total = len(records)
    exact, normalized = modes["exact"], modes["normalized_whitespace"]
    resolved = sum(record["location_resolved"] for record in records)
    unmatched = total - exact - normalized
    return {
        "total": total,
        "exact_matches": exact,
        "normalized_matches": normalized,
        "unmatched": unmatched,
        "hash_mismatch": modes["hash_mismatch"],
        "missing_document": modes["missing_document"],
        "invalid_evidence": modes["invalid_evidence"],
        "alias_source_mismatch": modes["alias_source_mismatch"],
        "ambiguous_matches": sum(r["status"] == "ambiguous" for r in records),
        "resolved_matches": resolved,
        "at_original_locator": sum(r["locator_status"] == "original" for r in records),
        "relocated_matches": sum(r["locator_status"] == "relocated" for r in records),
        "reference_view_matches": sum(r["reference_view"] and r["match_mode"] in {"exact", "normalized_whitespace"} for r in records),
        "coverage": {
            "found_fraction": (exact + normalized) / total if total else None,
            "exact_fraction": exact / total if total else None,
            "resolved_fraction": resolved / total if total else None,
        },
        "all_links_found": unmatched == 0 if total else None,
        "all_locations_resolved": resolved == total if total else None,
        "cases_without_evidence": [row["case_id"] for row in case_rows if not row["has_source_evidence"]],
        "case_results": case_rows,
        "records": records,
        "verification_scope": "새 파싱 원문의 해시·인용 복구·원문 위치 검증. 의미 지원 여부와 실제 RAG 답변 성능 평가는 별도.",
        "hash_basis": "로더가 새 원본 바이트에서 계산한 documents[].sha256과 gold 원본 해시를 비교함",
        "offset_convention": "정규화 전 block.text의 0 기반 Python 문자 인덱스, char_end 미포함",
    }
