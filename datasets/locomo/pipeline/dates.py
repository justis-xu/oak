"""确定性日期工具：英文会话日期解析、中文相对日期推算、答案归一化比较。

全部纯函数（零 LLM）——时间题的"计算日期"工具与判题预检共用此内核。
"""
from __future__ import annotations

import re
import unicodedata
from datetime import date, timedelta

# ---------------------------------------------------------------- 英文会话日期
_MONTHS_EN = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11,
    "december": 12,
}
_DT_RE = re.compile(
    r"(?:\d{1,2}:\d{2}\s*(?:am|pm)\s+on\s+)?(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})",
    re.I,
)


def parse_session_datetime(raw: str) -> date | None:
    """'1:56 pm on 8 May, 2023' / '8 May, 2023' → date。"""
    if not raw:
        return None
    m = _DT_RE.search(raw)
    if not m:
        return None
    day, mon_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
    mon = _MONTHS_EN.get(mon_name)
    if not mon:
        return None
    try:
        return date(year, mon, day)
    except ValueError:
        return None


# ---------------------------------------------------------------- 中文数字
_CN_DIGITS = {"零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
              "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
_CN_MONTH_WORDS = {"一月": 1, "二月": 2, "三月": 3, "四月": 4, "五月": 5, "六月": 6,
                   "七月": 7, "八月": 8, "九月": 9, "十月": 10, "十一月": 11, "十二月": 12,
                   "元月": 1, "正月": 1}


def cn_num(s: str) -> int | None:
    """中文数字（<100）→ int：'三'→3，'十'→10，'二十一'→21，'两'→2。"""
    s = s.strip()
    if not s:
        return None
    if s.isdigit():
        return int(s)
    if len(s) == 1:
        return _CN_DIGITS.get(s)
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGITS.get(left, 1) if left else 1
        ones = _CN_DIGITS.get(right, 0) if right else 0
        if (left and left not in _CN_DIGITS) or (right and right not in _CN_DIGITS):
            return None
        return tens * 10 + ones
    if all(c in _CN_DIGITS for c in s) and len(s) <= 4:   # 简单连写（一三 不常见，容忍）
        v = int("".join(str(_CN_DIGITS[c]) for c in s))
        return v if v < 32 else None
    return None


# ---------------------------------------------------------------- 星期与月份
_WEEKDAY_WORDS = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5,
                  "日": 6, "天": 6}


def _weekday_of(s: str) -> int | None:
    """'周日'/'星期三'/'礼拜天' → 0..6（周一=0）。"""
    m = re.search(r"[周星期礼拜]+\s*([一二三四五六日天])", s)
    return _WEEKDAY_WORDS.get(m.group(1)) if m else None


def _monday_of(d: date) -> date:
    return d - timedelta(days=d.weekday())


# ---------------------------------------------------------------- 显式中文日期
_DATE_FULL_RE = re.compile(
    r"(?:(\d{4})\s*年)?\s*(\d{1,2}|[一二三四五六七八九十]{1,3}|[一二]十[一二三四五六七八九]?)\s*月\s*"
    r"(?:(\d{1,2}|[一二三四五六七八九十]{1,3})\s*[日号])?"
)
_YEAR_ONLY_RE = re.compile(r"(\d{4})\s*年")
_CN_MONTH_RE = re.compile(r"([一二三四五六七八九十]{1,3})\s*月")


def parse_cn_date(s: str, default_year: int | None = None) -> tuple[int, int, int] | None:
    """解析 '2023年5月7日'/'5月7日'/'2023年6月'，返回 (y, m, d)，缺失部分为 0。
    无法解析返回 None。"""
    m = _DATE_FULL_RE.search(s)
    if not m:
        return None
    y = int(m.group(1)) if m.group(1) else 0
    mon = cn_num(m.group(2))
    day = cn_num(m.group(3)) if m.group(3) else 0
    if mon is None or not (1 <= mon <= 12):
        return None
    if day and not (1 <= day <= 31):
        return None
    if not y:
        y = default_year or 0
    return (y, mon, day)


# ---------------------------------------------------------------- 相对日期推算
def resolve_relative(anchor: date, expr: str) -> tuple[str, str]:
    """中文相对时间表达 → (iso日期 | YYYY-MM | YYYY | 区间'起~止', 粒度)。

    粒度 ∈ {日, 周, 月, 年, 无}。无法解析返回 ("", "无")。
    周约定：周一为一周之始（中国惯例）。
    """
    e = expr.strip()
    granularity = "日"

    # —— 纯年份：'2022' / '2022年'（核心解析不处理无月日的串，先短路）
    if re.fullmatch(r"(\d{4})\s*年?", e.replace(" ", "")):
        return (re.search(r"\d{4}", e).group(0), "年")

    # —— 组合表达："X之前的那个周日 / 那一周"
    m = re.search(r"(.+?)之前的那个?(周日|星期日|礼拜天|礼拜日)", e)
    if m:
        base = _resolve_core(anchor, m.group(1))
        if base:
            d = _last_weekday_before(base, 6)      # 周日=6
            return (d.isoformat(), "日")
    m = re.search(r"(.+?)之前的那个?(周|星期|礼拜|那?一周)", e)
    if m:
        base = _resolve_core(anchor, m.group(1))
        if base:
            start = base - timedelta(days=7)
            return (f"{start.isoformat()}~{(base - timedelta(days=1)).isoformat()}", "周")

    # 去年 / 今年 / 前年（年粒度）
    if "前年" in e:
        return (str(anchor.year - 2), "年")
    if "去年" in e:
        return (str(anchor.year - 1), "年")
    if "今年" in e:
        return (str(anchor.year), "年")

    # N 个月前/后、上上个月/上个月/下个月（月粒度）
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*(?:个)?月[前后]", e)
    if m:
        n = cn_num(m.group(1)) or 0
        if n < 200:
            d = _shift_month(anchor, -n if "前" in e else n)
            return (f"{d.year:04d}-{d.month:02d}", "月")
    if "上上个月" in e:
        d = _shift_month(anchor, -2)
        return (f"{d.year:04d}-{d.month:02d}", "月")
    if "上个月" in e:
        d = _shift_month(anchor, -1)
        return (f"{d.year:04d}-{d.month:02d}", "月")
    if "下个月" in e:
        d = _shift_month(anchor, 1)
        return (f"{d.year:04d}-{d.month:02d}", "月")

    # N 个周末前/后（周粒度）
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*个?周末[前后]", e)
    if m:
        n = cn_num(m.group(1)) or 0
        if 0 < n < 100:
            base = anchor - timedelta(weeks=n) if "前" in e else anchor + timedelta(weeks=n)
            return (base.isoformat(), "周")

    # 上周/这周/下周（整周，周粒度，取周一为锚）
    if re.fullmatch(r"(上|上上|这|本|下)?(周|星期|礼拜)", e.replace(" ", "")):
        base = _resolve_core(anchor, e)
        if base:
            return (base.isoformat(), "周")

    val = _resolve_core(anchor, e)
    if val is None:
        return ("", "无")
    # 粒度推断：显式年月无日 → 月；仅年 → 年
    if _YEAR_ONLY_RE.fullmatch(e.replace(" ", "")):
        granularity = "年"
        return (f"{val.year}", granularity)
    m = _DATE_FULL_RE.search(e)
    if m and not m.group(3) and m.group(1):
        return (f"{val.year:04d}-{val.month:02d}", "月")
    if m and not m.group(3) and not m.group(1):
        return (f"{val.month}月", "月")
    return (val.isoformat(), granularity)


def _resolve_core(anchor: date, e: str) -> date | None:
    """相对表达 → 具体 date（尽量）。返回 None 表示无法解析。"""
    e = e.strip()

    # 今天 / 昨天 / 前天 / 明天 / 后天
    if "今天" in e:
        return anchor
    if "大前天" in e:
        return anchor - timedelta(days=3)
    if "前天" in e:
        return anchor - timedelta(days=2)
    if "昨天" in e:
        return anchor - timedelta(days=1)
    if "大后天" in e:
        return anchor + timedelta(days=3)
    if "后天" in e:
        return anchor + timedelta(days=2)
    if "明天" in e:
        return anchor + timedelta(days=1)

    # 上上周X / 上周X / 这周X / 本周X / 下周X
    # 语义对齐数据集 gold（英语 "last Friday" 惯例）：上周X = 严格早于锚日的最近一个周X
    wd = _weekday_of(e)
    if wd is not None:
        if "上上周" in e:
            return _last_weekday_before(anchor, wd) - timedelta(days=7)
        if "上周" in e or "上星期" in e or "上礼拜" in e:
            return _last_weekday_before(anchor, wd)
        this_mon = _monday_of(anchor)
        if "下周" in e or "下星期" in e or "下礼拜" in e:
            mon = this_mon + timedelta(days=7)
        else:                                        # 这周/本周/周X
            mon = this_mon
        return mon + timedelta(days=wd)

    # 上上个月 / 上个月 / 这个月/本月 / 下个月
    if "上上个月" in e:
        return _shift_month(anchor, -2)
    if "上个月" in e:
        return _shift_month(anchor, -1)
    if "下个月" in e:
        return _shift_month(anchor, 1)

    # N 个星期/周/月/年 前/后
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*(?:个)?(星期|周|礼拜)[前后]", e)
    if m:
        n = cn_num(m.group(1)) or 0
        delta = timedelta(weeks=n) if n < 200 else None
        if delta is not None:
            return anchor - delta if "前" in m.group(3) else anchor + delta
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*(?:个)?月[前后]", e)
    if m:
        n = cn_num(m.group(1)) or 0
        if n < 200:
            return _shift_month(anchor, -n if "前" in e else n)
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*(?:个)?年[前后]", e)
    if m:
        n = cn_num(m.group(1)) or 0
        if n < 200:
            try:
                return date(anchor.year - n if "前" in e else anchor.year + n,
                            anchor.month, anchor.day)
            except ValueError:
                return None
    m = re.search(r"([0-9一二两三四五六七八九十]+)\s*天[前后]", e)
    if m:
        n = cn_num(m.group(1)) or 0
        if n < 10000:
            return anchor - timedelta(days=n) if "前" in e else anchor + timedelta(days=n)

    # 去年 / 今年 / 前年
    if "前年" in e:
        return date(anchor.year - 2, anchor.month, anchor.day)
    if "去年" in e:
        return date(anchor.year - 1, anchor.month, anchor.day)
    if "今年" in e:
        return anchor

    # 上周（整周）/ 这周 / 下周 —— 取该周周一
    if e in ("上周", "上星期", "上礼拜"):
        return _monday_of(anchor) - timedelta(days=7)
    if e in ("这周", "本周", "这星期", "本星期"):
        return _monday_of(anchor)
    if e in ("下周", "下星期", "下礼拜"):
        return _monday_of(anchor) + timedelta(days=7)

    # 显式日期
    ymd = parse_cn_date(e, default_year=anchor.year)
    if ymd and ymd[0]:
        try:
            return date(*ymd) if ymd[2] else date(ymd[0], ymd[1], 1)
        except ValueError:
            return None
    if ymd and ymd[1]:                                # 只有 X月X日，无年
        try:
            return date(anchor.year, ymd[1], ymd[2] or 1)
        except ValueError:
            return None

    return None


def _shift_month(d: date, k: int) -> date:
    """月份平移（日溢出夹到月末）。"""
    idx = d.year * 12 + (d.month - 1) + k
    y, m = divmod(idx, 12)
    m += 1
    last = _days_in_month(y, m)
    return date(y, m, min(d.day, last))


def _days_in_month(y: int, m: int) -> int:
    if m == 12:
        nxt = date(y + 1, 1, 1)
    else:
        nxt = date(y, m + 1, 1)
    return (nxt - timedelta(days=1)).day


def _last_weekday_before(base: date, wd: int) -> date:
    """严格早于 base 的最近一个周 wd（wd: 0=周一..6=周日）。"""
    d = base - timedelta(days=1)
    while d.weekday() != wd:
        d -= timedelta(days=1)
    return d


# ---------------------------------------------------------------- 答案归一化
_PUNCT_RE = re.compile(r"[\s，。、；;：:！!？?\.\,\(\)（）\"'“”‘’\[\]【】~—]+")


def normalize_answer_text(s: str) -> str:
    """判题预检的归一化：NFKC→中文数字月→年月日规范化→剥标点空白。"""
    if s is None:
        return ""
    t = unicodedata.normalize("NFKC", str(s))

    def _mon_repl(m):
        v = cn_num(m.group(1))
        return f"{v}月" if v else m.group(0)

    t = _CN_MONTH_RE.sub(_mon_repl, t)
    # 中文数字日：X日/X号（≤三十一）
    t = re.sub(r"([一二三四五六七八九十]{1,3})\s*[日号]",
               lambda m: (f"{cn_num(m.group(1))}日"
                          if cn_num(m.group(1)) is not None else m.group(0)),
               t)
    # 年月日 → y-m-d（保留可缺省成分）
    t = re.sub(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?",
               r"\1-\2-\3", t)
    t = re.sub(r"(\d{4})\s*年\s*(\d{1,2})\s*月", r"\1-\2", t)
    t = re.sub(r"(\d{4})\s*年", r"\1", t)
    t = re.sub(r"(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]?", r"\1-\2", t)
    t = re.sub(r"(\d{1,2})\s*月", r"\1月", t)
    return _PUNCT_RE.sub("", t).lower()


def extract_digits(s: str) -> str:
    return "".join(re.findall(r"\d+", str(s or "")))


def date_components(s: str) -> tuple[int, int, int] | None:
    """从归一化答案中解析 (y, m, d)（缺失为 0）。仅接受日期形态。"""
    t = str(s or "").strip()
    m = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})", t)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", t)
    if m:
        return int(m.group(1)), int(m.group(2)), 0
    m = re.fullmatch(r"(\d{4})", t)
    if m:
        return int(m.group(1)), 0, 0
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", t)
    if m:
        return 0, int(m.group(1)), int(m.group(2))
    m = re.fullmatch(r"(\d{1,2})月", t)
    if m:
        return 0, int(m.group(1)), 0
    return None


def answer_equivalent(gold: str | int, pred: str) -> bool:
    """确定性等价判断（判题预检用，只短路明显 exact，拿不准返回 False 交 LLM）。"""
    if pred is None:
        return False
    g = normalize_answer_text(gold)
    p = normalize_answer_text(pred)
    if not g or not p:
        return False
    if g == p:
        return True
    # 纯数字等价：gold=2022 / pred="2022年"
    gd, pd_ = extract_digits(g), extract_digits(p)
    if gd and pd_ and gd == pd_ and re.fullmatch(r"\d+", g):
        return pd_.startswith(gd) and len(re.sub(r"\D", "", p)) == len(gd)
    # 日期等价：gold 的各成分都被 pred 覆盖且相等（pred 可更细）
    gc, pc = date_components(g), date_components(p)
    if gc and pc:
        for gv, pv in ((gc[0], pc[0]), (gc[1], pc[1]), (gc[2], pc[2])):
            if gv and pv and gv != pv:
                return False
            if gv and not pv:
                return False            # gold 有年份 pred 没有 → 交给 LLM
        return True
    # 多元素列表（、/,/;分隔）集合等价（顺序无关，元素须逐一完全一致）
    gs = {normalize_answer_text(x) for x in re.split(r"[、,，;；]", str(gold)) if x.strip()}
    ps = {normalize_answer_text(x) for x in re.split(r"[、,，;；]", str(pred)) if x.strip()}
    if gs and ps and gs == ps and len(gs) > 0:
        return not gs & {""}            # 元素皆非空
    return False
