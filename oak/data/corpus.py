"""语料分块：reference_information（每题专属语料）→ chunk 列表。

分块策略：每个源的 DataFrame repr 按**整行**切，表头行复制进每个 chunk——
抽取器每块都能看到原始字段名（大小写与语料逐字一致）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

SOURCE_KEYS = {
    "flights": "Flight",
    "accommodations": "Accommodation",
    "restaurants": "Restaurant",
    "attractions": "Attraction",
    "distances": "Distance",
}


@dataclass
class Chunk:
    source: str          # flights | accommodations | restaurants | attractions | distances
    seq: int             # 该源内的块序号
    header: str          # 表头行（含原始字段名）
    text: str            # 数据行

    def render(self) -> str:
        return f"[source: {self.source}]\n{self.header}\n{self.text}"


_ROW_RE = re.compile(r"\n(?=\d+[, ])")     # DataFrame repr 的行首（索引, 字段...）


def _split_rows(content: str) -> tuple[str, list[str]]:
    """把 DataFrame repr 分成表头和数据行。"""
    content = content.strip()
    lines = content.split("\n")
    if not lines:
        return "", []
    # 表头：第一行（可能前缀是索引列名如 ",NAME,room type,..."）
    header = lines[0]
    rows = [ln for ln in lines[1:] if ln.strip()]
    return header, rows


def reference_chunks(ref_info: dict[str, str], max_chars: int = 5000) -> list[Chunk]:
    chunks: list[Chunk] = []
    for desc, content in ref_info.items():
        source = _match_source(desc)
        if source is None:
            continue
        header, rows = _split_rows(content)
        batch: list[str] = []
        size = 0
        seq = 0
        for row in rows:
            batch.append(row)
            size += len(row) + 1
            if size >= max_chars:
                chunks.append(Chunk(source, seq, header, "\n".join(batch)))
                seq += 1
                batch, size = [], 0
        if batch:
            chunks.append(Chunk(source, seq, header, "\n".join(batch)))
    return chunks


def _match_source(desc: str) -> str | None:
    d = desc.lower()
    if "flight" in d:
        return "flights"
    if "accommodation" in d or "hotel" in d or "airbnb" in d:
        return "accommodations"
    if "restaurant" in d:
        return "restaurants"
    if "attraction" in d:
        return "attractions"
    return None


def distance_chunks(tp_root: Path) -> list[Chunk]:
    """城际地面交通（官方环境数据），作为附加语料。"""
    f = tp_root / "database" / "googleDistanceMatrix" / "distance.csv"
    if not f.exists():
        return []
    lines = f.read_text().splitlines()
    if not lines:
        return []
    header = lines[0]
    rows = lines[1:]
    out: list[Chunk] = []
    max_rows = 200
    for seq, i in enumerate(range(0, len(rows), max_rows)):
        out.append(Chunk("distances", seq, header, "\n".join(rows[i:i + max_rows])))
    return out
