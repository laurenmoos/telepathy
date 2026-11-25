"""Programs in ordinary Python, compiled down to the instructions in lang.py (run by run.py).

Words look like four lists. Reading one gives its value on the current cell; assigning to one
moves the model's state so the word reads that value:
    answer[2] = 3.0
    answer[0] = 0.5 * row_end[3] - col_end[7]
    answer[1] = 0                               # back to typical
Supported Python:
    numbers and variables; words in sums, with constant factors (*, /, unary -)
    if / elif / else on comparisons (>, <, >=, <=), joined with and / or / not
    best = argmax(w0, w1, ...) or argmax([w0, w1, ...]), then if best == k / elif best == k
    for loops over range(...) or a list, unrolled when compiled; pass
Everything is decided per cell while the model runs, so word values are not known when the
program is compiled: they can only be compared, summed, scaled and passed to argmax.
"""
import ast

from . import lang


class CompileError(ValueError):
    pass


class Lin:
    """A sum of words with constant factors, plus a constant: {word id: factor}, const."""

    def __init__(self, terms=None, const=0.0):
        self.terms, self.const = dict(terms or {}), float(const)

    def __add__(self, o):
        t = dict(self.terms)
        for w, c in o.terms.items():
            t[w] = t.get(w, 0.0) + c
        return Lin({w: c for w, c in t.items() if c != 0}, self.const + o.const)

    def scale(self, k):
        return Lin({w: c * k for w, c in self.terms.items()}, self.const * k)


class Choice:
    """The result of argmax: which word is largest on this cell. Usable only in if best == k."""

    def __init__(self, words):
        self.words, self.switch = words, None


GROUPS = ("row_end", "col_end", "box_end", "answer")


def compile_python(source):
    """Python source -> lang.Prog."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        raise CompileError(f"line {e.lineno}: {e.msg}")
    return lang.Prog(_block(tree.body, {}))


def _err(node, msg):
    raise CompileError(f"line {getattr(node, 'lineno', '?')}: {msg}")


# ---------------------------------------------------------------- values at compile time

def _value(node, env):
    """A compile-time value (int, float, list, Choice) or a Lin over words."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in env:
            return env[node.id]
        if node.id in GROUPS:
            _err(node, f"{node.id} is a list of words; use one word, like {node.id}[0]")
        _err(node, f"{node.id} is not defined")
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id in GROUPS:
        return Lin({_word(node, env): 1.0})
    if isinstance(node, ast.Subscript):
        seq, i = _value(node.value, env), _value(node.slice, env)
        if isinstance(seq, list) and isinstance(i, int):
            if not -len(seq) <= i < len(seq):
                _err(node, f"index {i} is out of range")
            return seq[i]
        _err(node, "only lists can be indexed, with a whole number")
    if isinstance(node, (ast.List, ast.Tuple)):
        return [_value(x, env) for x in node.elts]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        v = _value(node.operand, env)
        return _neg(v, node) if isinstance(node.op, ast.USub) else v
    if isinstance(node, ast.BinOp):
        a, b = _value(node.left, env), _value(node.right, env)
        num = (int, float)
        if isinstance(a, num) and isinstance(b, num):
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                   ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b}
            if type(node.op) not in ops:
                _err(node, "this operator is not supported")
            return ops[type(node.op)]()
        a, b = _lin(a, node), _lin(b, node)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a + b.scale(-1)
        if isinstance(node.op, ast.Mult):
            if not a.terms:
                return b.scale(a.const)
            if not b.terms:
                return a.scale(b.const)
            _err(node, "two word values cannot be multiplied; only a word times a number")
        if isinstance(node.op, ast.Div) and not b.terms:
            return a.scale(1 / b.const)
        _err(node, "words can only be added, subtracted, and multiplied or divided by a number")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        f = node.func.id
        args = [_value(x, env) for x in node.args]
        if f == "range" and all(isinstance(x, int) for x in args) and 1 <= len(args) <= 3:
            return list(range(*args))
        if f == "argmax":
            if len(args) == 1 and isinstance(args[0], list):
                args = args[0]
            words = []
            for x in args:
                if not (isinstance(x, Lin) and len(x.terms) == 1 and x.const == 0 and list(x.terms.values()) == [1.0]):
                    _err(node, "argmax takes single words, like argmax(row_end[0], row_end[4])")
                words.append(next(iter(x.terms)))
            if not 2 <= len(words) <= 9 or len(set(words)) != len(words):
                _err(node, "argmax takes 2 to 9 different words")
            return Choice(words)
        if f == "len" and len(args) == 1 and isinstance(args[0], list):
            return len(args[0])
        if f in ("abs", "min", "max"):
            _err(node, f"{f}() of word values is not supported; use if or argmax")
        _err(node, f"{f}() is not supported")
    _err(node, f"this kind of expression is not supported: {ast.unparse(node)}")


def _word(node, env):
    i = _value(node.slice, env)
    group = node.value.id
    if not isinstance(i, int):
        _err(node, f"a word index must be a whole number, not {ast.unparse(node.slice)}")
    name = f"{group}[{i}]"
    if name not in lang.NAME_TO_WORD:
        _err(node, f"there is no word {name} ({group} has {lang.GROUP_SIZE.get(group, 0)} words, from 0)")
    return lang.NAME_TO_WORD[name]


def _lin(v, node):
    if isinstance(v, Lin):
        return v
    if isinstance(v, (int, float)):
        return Lin({}, v)
    _err(node, "expected a number or word values here")


def _neg(v, node):
    return -v if isinstance(v, (int, float)) else _lin(v, node).scale(-1)


# ---------------------------------------------------------------- code

def _load(lin, node):
    """Instructions that leave a sum of words (no constant) in acc."""
    terms = sorted(lin.terms.items(), key=lambda t: lang.WORD_LAYERS[t[0]])     # earlier layers first
    if not terms:
        return [lang.Const(0.0)]
    out = [lang.Op("load", terms[0][0])]                  # Horner form, as lang.compile_expr
    for (_, a), (w, b) in zip(terms, terms[1:]):
        out += [lang.Scale(round(a / b, 6)), lang.Op("add", w)]
    if terms[-1][1] != 1:
        out.append(lang.Scale(round(terms[-1][1], 6)))
    return out


def _store(word, v, node):
    if isinstance(v, (int, float)):
        return [lang.Const(float(v)), lang.Op("store", word)]
    lin = _lin(v, node)
    if lin.terms and lin.const:
        _err(node, "a word can be set to a number or to a sum of words, not both at once")
    if not lin.terms:
        return [lang.Const(lin.const), lang.Op("store", word)]
    return _load(lin, node) + [lang.Op("store", word)]


def _cond(node, env, then, other):
    """Instructions for: if node: then else: other."""
    if isinstance(node, ast.BoolOp):
        first, rest = node.values[0], node.values[1:]
        rest_node = rest[0] if len(rest) == 1 else ast.BoolOp(op=node.op, values=rest)
        if isinstance(node.op, ast.And):
            return _cond(first, env, _cond(rest_node, env, then, other), other)
        return _cond(first, env, then, _cond(rest_node, env, then, other))
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return _cond(node.operand, env, other, then)
    if isinstance(node, ast.Compare) and len(node.ops) == 1:
        a, b = _value(node.left, env), _value(node.comparators[0], env)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):        # known when compiled
            ok = {ast.Gt: a > b, ast.Lt: a < b, ast.GtE: a >= b, ast.LtE: a <= b,
                  ast.Eq: a == b, ast.NotEq: a != b}.get(type(node.ops[0]))
            if ok is None:
                _err(node, "this comparison is not supported")
            return then if ok else other
        op = node.ops[0]
        if isinstance(op, (ast.Lt, ast.LtE)):
            a, b, op = b, a, ast.Gt()
        if not isinstance(op, (ast.Gt, ast.GtE)):
            _err(node, "word values can be compared with >, <, >= or <= only")
        d = _lin(a, node) + _lin(b, node).scale(-1)
        if not d.terms:
            _err(node, "this comparison has no word in it")
        return _load(Lin(d.terms), node) + [lang.If(-d.const, then, other)]
    if isinstance(node, ast.Constant) and isinstance(node.value, bool):
        return then if node.value else other
    _err(node, f"an if needs a comparison, like row_end[3] > 0.5, not {ast.unparse(node)}")


def _choice_test(node, env):
    """(choice, k) if node is `best == k` for an argmax result, else None."""
    if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
        a = _value(node.left, env) if isinstance(node.left, ast.Name) else None
        if isinstance(a, Choice):
            k = _value(node.comparators[0], env)
            if not isinstance(k, int) or not 0 <= k < len(a.words):
                _err(node, f"best == {ast.unparse(node.comparators[0])}: argmax has {len(a.words)} words, numbered from 0")
            return a, k
    return None


def _block(body, env):
    out = []
    for node in body:
        out += _stmt(node, env)
    return out


def _stmt(node, env):
    if isinstance(node, ast.Pass):
        return []
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
        return []                                                         # a docstring
    if isinstance(node, (ast.Assign, ast.AugAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if len(targets) != 1:
            _err(node, "assign one thing at a time")
        t = targets[0]
        v = _value(node.value, env)
        if isinstance(node, ast.AugAssign):
            v = _value(ast.BinOp(left=t, op=node.op, right=node.value), env)
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Name) and t.value.id in GROUPS:
            if isinstance(v, (list, Choice)):
                _err(node, "a word can be set to a number or a sum of words")
            return _store(_word(t, env), v, node)
        if isinstance(t, ast.Name):
            if t.id in GROUPS:
                _err(node, f"{t.id} cannot be replaced")
            if isinstance(v, Choice):
                v = Choice(v.words)
                sw = lang.Switch(list(v.words), [[] for _ in v.words])
                v.switch = sw
                env[t.id] = v
                return [sw]
            env[t.id] = v
            return []
        _err(node, "only a variable or a word can be assigned to")
    if isinstance(node, ast.If):
        test = _choice_test(node.test, env)
        if test:                                   # if best == k / elif best == j / ... / else
            choice, done, n = test[0], [], node
            while True:
                t = _choice_test(n.test, env)
                if not t or t[0] is not choice:
                    _err(n, "an if on an argmax result can only be followed by elif on the same result")
                choice.switch.branches[t[1]] += _block(n.body, env)
                done.append(t[1])
                if len(n.orelse) == 1 and isinstance(n.orelse[0], ast.If):
                    n = n.orelse[0]
                    continue
                if n.orelse:
                    rest = _block(n.orelse, env)
                    for k in range(len(choice.words)):
                        if k not in done:
                            choice.switch.branches[k] += rest
                return []
        return _cond(node.test, env, _block(node.body, env), _block(node.orelse, env))
    if isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name):
            _err(node, "a for loop needs one loop variable")
        seq = _value(node.iter, env)
        if not isinstance(seq, list):
            _err(node, "a for loop must go over range(...) or a list")
        if node.orelse:
            _err(node, "for ... else is not supported")
        out = []
        for x in seq:
            env[node.target.id] = x
            out += _block(node.body, env)
        return out
    if isinstance(node, (ast.FunctionDef, ast.While, ast.Return, ast.Import, ast.ImportFrom)):
        _err(node, f"{type(node).__name__.lower()} is not supported")
    _err(node, f"this statement is not supported: {ast.unparse(node).splitlines()[0]}")
