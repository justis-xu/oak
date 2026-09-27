"""函数沙箱：AST 白名单检查 + 受限 exec + 真图试跑。"""
from __future__ import annotations

import ast
import concurrent.futures
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import networkx as nx

from . import library as ops

ALLOWED_BUILTINS = {
    "len", "min", "max", "sum", "sorted", "set", "dict", "list", "tuple",
    "str", "int", "float", "round", "abs", "enumerate", "zip", "range", "bool",
    "any", "all", "isinstance", "ceil",
}
ALLOWED_CALLS = set(ops.OPERATOR_REGISTRY.keys())
# 常用字符串/容器方法放行（含本地容器的写方法——无 IO 风险；下划线属性仍然禁止）
ALLOWED_METHODS = {
    "lower", "upper", "strip", "lstrip", "rstrip", "split", "rsplit", "splitlines",
    "join", "startswith", "endswith", "replace", "format", "title", "capitalize",
    "isdigit", "isalpha", "items", "keys", "values", "get", "copy",
    "append", "extend", "insert", "add", "update", "pop", "remove",
}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal)


class SandboxError(ValueError):
    pass


# ---------------------------------------------------------------- AST 检查
def _check_node(node: ast.AST) -> None:
    if isinstance(node, FORBIDDEN_NODES):
        raise SandboxError(f"forbidden statement: {type(node).__name__}")
    if isinstance(node, ast.Attribute):
        if node.attr.startswith("_"):
            raise SandboxError(f"forbidden attribute access: _{node.attr}")
    if isinstance(node, ast.Name):
        if node.id.startswith("__"):
            raise SandboxError(f"forbidden dunder name: {node.id}")
    if isinstance(node, ast.Call):
        fn = node.func
        name = None
        if isinstance(fn, ast.Name):
            name = fn.id
        elif isinstance(fn, ast.Attribute):
            name = fn.attr
        if name and name not in ALLOWED_CALLS and name not in ALLOWED_BUILTINS \
                and name not in ALLOWED_METHODS:
            raise SandboxError(
                f"call to {name!r} not allowed (only the nine operators + "
                f"builtins {sorted(ALLOWED_BUILTINS)} + safe methods "
                f"{sorted(ALLOWED_METHODS)})")
    for child in ast.iter_child_nodes(node):
        _check_node(child)


def check_source(src: str) -> str:
    """静态检查；返回函数名（单函数文件）。"""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise SandboxError(f"SyntaxError: {e}")
    _check_node(tree)
    fn_names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if not fn_names:
        raise SandboxError("no function definition found")
    return fn_names[0]


# ---------------------------------------------------------------- exec 与试跑
def safe_exec_namespace() -> dict[str, Any]:
    """共享受限命名空间：九算子 + math.ceil + 白名单 builtins。

    沙箱试跑（exec_function_source）与正式运行（FunctionCatalog.module）必须共用，
    否则会出现"试跑通过、正式调用 NameError: ceil"（q5/q66 实证）。
    """
    import math
    ns: dict[str, Any] = {name: getattr(ops, name) for name in ops.OPERATOR_REGISTRY}
    real = __builtins__ if isinstance(__builtins__, dict) else getattr(__builtins__, "__dict__", {})
    blts = {b: real[b] for b in ALLOWED_BUILTINS if b != "ceil" and b in real}
    blts["ceil"] = math.ceil            # ceil 非内置，恒映射 math.ceil
    ns["__builtins__"] = blts
    ns["ceil"] = math.ceil
    return ns


def exec_function_source(src: str) -> tuple[str, Callable]:
    """AST 白名单 → 受限命名空间 exec。返回 (fn_name, fn)。"""
    fn_name = check_source(src)
    ns = safe_exec_namespace()
    exec(compile(src, "<generated>", "exec"), ns)          # noqa: S102 —— 白名单前置
    return fn_name, ns[fn_name]


@dataclass
class TrialResult:
    graph_name: str
    ok: bool
    error: str | None
    sample_output: Any
    elapsed_ms: int


def _run_with_timeout(fn: Callable, args: dict, timeout_s: float, g) -> Any:
    def wrapped():
        ops.set_graph(g)               # 工作线程不继承协程 ContextVar，需线程内注入
        return fn(**args)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(wrapped)
        return fut.result(timeout=timeout_s)


def trial_run(fn: Callable, graphs: dict[str, nx.MultiDiGraph],
              sample_args: dict, timeout_s: float = 10.0) -> list[TrialResult]:
    """在每个图上试跑一次（真图 + 单题 mini 图）。图注入发生在工作线程内。"""
    results = []
    for gname, g in graphs.items():
        t0_ms = __import__("time").perf_counter() * 1000
        try:
            out = _run_with_timeout(fn, sample_args, timeout_s, g)
            import json as _json
            _json.dumps(out)                    # 可序列化检查
            results.append(TrialResult(gname, True, None, _json.dumps(out)[:300],
                                       int(__import__("time").perf_counter() * 1000 - t0_ms)))
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            results.append(TrialResult(gname, False, f"{type(e).__name__}: {e}\n{tb}",
                                       None, int(__import__("time").perf_counter() * 1000 - t0_ms)))
    return results


def check_hardcoded_fields(src: str, schema) -> list[str]:
    """函数源码中硬编码的 schema 字段引用是否存在于模式（R11 预防）。"""
    import re
    from ..schema.model import Schema
    strings = re.findall(r'["\']([^"\']+)["\']', src)
    known_types = {e.name for e in schema.entities}
    known_fields: set[str] = set()
    for e in schema.entities:
        known_fields |= e.attr_names()
    known_rels = {r.name for r in schema.relations}
    bad = []
    for s in strings:
        # 只检查像字段名的字符串（含空格小写词或驼峰），排除操作名/关系名
        if s in known_rels or s in OPERATOR_CALLS_SAFE:
            continue
        if s in known_types:
            continue
        # 若字符串形如 "xx yy"（含空格、全字母）且是属性风格却不在 schema —— 可能是字段拼错
        if " " in s and re.fullmatch(r"[A-Za-z][A-Za-z0-9 _]*", s):
            # 宽松：只有当它接近某个已知字段（编辑距离 1 内的大小写差异）才报
            low = s.lower()
            if low not in {f.lower() for f in known_fields}:
                bad.append(s)
    return bad


OPERATOR_CALLS_SAFE = frozenset(ops.OPERATOR_REGISTRY.keys())
