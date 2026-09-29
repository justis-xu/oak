"""函数目录：published/failed 状态、源码拼装、渲染给 ReAct、落盘。"""
from __future__ import annotations

import json
import types
from dataclasses import dataclass, field, asdict
from pathlib import Path

from oak.operators import library as ops
from oak.operators.sandbox import safe_exec_namespace


@dataclass
class CompiledFunction:
    name: str
    signature: str              # 签名行文本
    docstring: str              # 首行语义
    source: str                 # 完整 python 源
    status: str = "published"   # published | failed
    trials: list = field(default_factory=list)
    generation: int = 0         # 被评判器 modify 后 +1

    def render(self) -> str:
        return f"- {self.signature}\n  {self.docstring}"


class FunctionCatalog:
    def __init__(self, functions: list[CompiledFunction] | None = None):
        self.functions: list[CompiledFunction] = functions or []

    def published(self) -> list[CompiledFunction]:
        return [f for f in self.functions if f.status == "published"]

    def module(self, g) -> types.ModuleType:
        """全部 published 源码拼成模块；图由调用方在题的协程上下文里 set_graph。"""
        mod = types.ModuleType("oak_generated_functions")
        ns: dict = safe_exec_namespace()      # 与沙箱试跑一致（ceil 等在正式运行同样可用）
        src = "\n\n".join(f.source for f in self.published())
        exec(compile(src, "<functions>", "exec"), ns)         # noqa: S102
        for k, v in ns.items():
            if callable(v) and not k.startswith("_"):
                setattr(mod, k, v)
        mod.G = g
        return mod

    def render_catalog(self) -> str:
        lines = []
        for f in self.published():
            # 例调用行：docstring 的 Example 段若有则一并展示
            lines.append(f"### {f.signature}")
            lines.append(f.docstring)
            ex = self._example_of(f)
            if ex:
                lines.append(f"example: {ex}")
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _example_of(f: CompiledFunction) -> str:
        for ln in f.docstring.splitlines():
            if ln.lower().lstrip().startswith("example"):
                return ln.split(":", 1)[-1].strip()
        # 从源码中抓 docstring 内的 Example 行
        import re
        m = re.search(r"Example:\s*(.+)", f.source)
        return m.group(1).strip() if m else ""

    # ---------- 落盘 ----------
    def save(self, dir: Path) -> None:
        dir.mkdir(parents=True, exist_ok=True)
        (dir / "functions.py").write_text(
            "\n\n".join(f.source for f in self.published()))
        (dir / "functions.json").write_text(json.dumps(
            [asdict(f) for f in self.functions], ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, dir: Path) -> "FunctionCatalog":
        data = json.loads((dir / "functions.json").read_text())
        fns = [CompiledFunction(**d) for d in data]
        return cls(fns)
