"""A model writes a program by calling two tools.

The writer (an LLM) is the agent. It sees the program language and the word names, and calls
(every text it sees is in config.yaml under prompts.agent):
    run(code, first, count)   run statements inside the steered model on the practice cells;
                              returns one raw row per cell: the model's answer, the right
                              answer, and the trace of what each statement did
    submit(code)              the final program; it is checked on the practice and held-out boards
The harness only executes calls and returns their output. It ends at submit or when the
writer runs out of calls.

Today the steered model is Qwen3-0.6B and the writer is another model. Later the writer will
query its own activations.

    telepathy/run.sh agent [--set agent.calls=40] [--out runs/agent]
"""
import json, os, random, re

from .programs import check, lang
from .programs.run import run
from .setup.log import log
from .setup.settings import settings, load_model, load_words
from .sudoku.boards import make_boards

def tools(p):
    """The two tools, described with the texts in config.yaml (prompts.agent)."""
    def tool(name, props, required):
        return {"type": "function", "function": {"name": name, "description": p[name], "parameters": {
            "type": "object", "properties": props, "required": required}}}
    return [tool("run", {"code": {"type": "string", "description": p["run_code"]},
                         "first": {"type": "integer", "description": p["run_first"]},
                         "count": {"type": "integer", "description": p["run_count"]}}, ["code"]),
            tool("submit", {"code": {"type": "string", "description": p["submit_code"]}}, ["code"])]


class Cells:
    """The practice cells, and running code on them with the full trace."""

    def __init__(self, m, boards, n, rng):
        cells = [(b["board"], r, c, b["solution"][r][c]) for b in boards for r in range(9) for c in range(9)
                 if b["board"][r][c] == 0]
        self.m, self.cells = m, rng.sample(cells, min(n, len(cells)))
        m.prepare([c[:3] for c in self.cells])

    def run(self, code, first=0, count=24):
        statements = lang.split_statements(code) if code.strip() else []
        prog = lang.compile_program(statements)
        if not lang.valid(prog):
            raise ValueError("the program is too long")
        out = run(self.m, [(*c[:3], prog) for c in self.cells])
        right = sum(d == c[3] for (d, _, _), c in zip(out, self.cells))
        rows = [f"cell {i}: answered {d}, right {c[3]}" + (f"; {'; '.join(tr)}" if tr else "")
                for i, ((d, tr, _), c) in enumerate(zip(out, self.cells))][first:first + count]
        return f"{right} of {len(self.cells)} cells right.\n" + "\n".join(rows), right


class Writer:
    """An MLX model that writes with thinking and native tool calls."""

    def __init__(self, path, a, tools):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.wrap = load(path)
        self.sampler = make_sampler(temp=a["temperature"], top_p=0.95, top_k=20)
        self.max_tokens, self.tools = a["max_tokens"], tools

    def reply(self, chat):
        from mlx_lm import stream_generate
        prompt = self.wrap.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True,
                                               enable_thinking=True)
        text = ""
        for r in stream_generate(self.model, self.wrap, prompt, max_tokens=self.max_tokens, sampler=self.sampler):
            text += r.text
            if "</tool_call>" in text:
                break
        return text


def tool_calls(text):
    out = []
    for body in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.S):
        try:
            call = json.loads(body)
            out.append((call["name"], call.get("arguments", {})))
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            out.append(("error", f"could not read the tool call: {e}"))
    return out


def main(argv=None):
    cfg, a = settings(argv, extra=lambda ap: ap.add_argument("--out", default="runs/agent"))
    ac = cfg["agent"]
    os.makedirs(a.out, exist_ok=True)
    m = load_model(cfg, a)
    load_words(m, cfg)
    practice_boards, test_boards = make_boards(cfg, log=log)
    cells = Cells(m, practice_boards, ac["cells"], random.Random(ac["seed"]))
    _, plain = cells.run("")
    log(f"no program: {plain} of {len(cells.cells)} practice cells right")

    names = ", ".join(f"{g}[0] to {g}[{n - 1}]" for g, n in lang.GROUP_SIZE.items())
    layers = sorted({L for L in lang.WORD_LAYERS.values()})
    answer_layers = ", ".join(f"{n} layer {lang.WORD_LAYERS[w]}" for n, w in lang.NAME_TO_WORD.items()
                              if n.startswith("answer"))
    p = cfg["prompts"]["agent"]
    system = p["system"].rstrip().format(words=names, cells=len(cells.cells), right=plain, calls=ac["calls"],
                           layers=f" row_end, col_end and box_end are at layer {layers[0]}; {answer_layers}.")
    chat = [{"role": "system", "content": system}, {"role": "user", "content": p["first"]}]
    writer = Writer(cfg["models"].get(ac["writer"], ac["writer"]), ac, tools(p))
    submitted, log = None, []
    for call in range(1, ac["calls"] + 1):
        text = writer.reply(chat)
        thought, _, said = text.rpartition("</think>") if "</think>" in text else (text, "", "")
        calls = tool_calls(said)
        chat.append({"role": "assistant", "content": said.strip()})
        log.append(dict(call=call, thinking=thought.replace("<think>", "").strip(), reply=said.strip(), results=[]))
        if not calls:
            chat.append({"role": "user", "content": p["no_tool"]})
            log(f"call {call}: no tool call")
            continue
        for name, args in calls:
            try:
                if name == "run":
                    result, right = cells.run(args.get("code", ""), int(args.get("first", 0)), int(args.get("count", 24)))
                    log(f"call {call}: run, {right} right\n    " + args.get("code", "").replace("\n", "\n    "))
                elif name == "submit":
                    submitted = lang.split_statements(args["code"])
                    result = "submitted"
                else:
                    result = args if name == "error" else f"there is no tool {name}"
            except (ValueError, KeyError, IndexError) as e:
                result = f"error: {e}"
                log(f"call {call}: {name}, error: {e}")
            chat.append({"role": "tool", "content": result})
            log[-1]["results"].append(result)
        json.dump(dict(system=system, log=log), open(f"{a.out}/session.json", "w"), indent=1)
        if submitted is not None:
            break

    if submitted is None:
        log("no program submitted")
        return
    open(f"{a.out}/program.txt", "w").write("\n".join(submitted) + "\n")
    log("submitted:\n" + lang.show(submitted))
    log("check:\n" + check.text(check.check(m, submitted, practice_boards, test_boards)))


if __name__ == "__main__":
    main()
