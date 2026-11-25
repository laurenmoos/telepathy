"""A model writes a program in Python, in a workspace of files, by calling tools.

The writer (an LLM) is the agent. Every text it sees, including the tool descriptions, is in
config.yaml under prompts.agent. Its workspace is <out>/workspace, with two files:
    program.py   the program (Python, compiled by programs/python.py)
    notes.md     its own notes; its thinking is not kept between calls, so this is its memory
    best.py      kept by the harness: the code with the most practice cells right so far
Tools:
    read_file(path), write_file(path, content)   the two workspace files
    run(code, first, count)    run code (or program.py, when code is empty) inside the steered
                               model on the practice cells; returns raw rows per cell: the
                               model's answer, the right answer, and the trace
    submit()                   end the session
The harness only executes calls and returns their output. It ends at submit or when the
writer runs out of calls; then best.py is checked on the practice and held-out boards.
Each reply has a thinking budget; when it runs out, the thinking is closed and the writer
gives its tool call.

Today the steered model is Qwen3-0.6B and the writer is another model. Later the writer will
query its own activations.

    telepathy/run.sh agent [--set agent.calls=40] [--out runs/agent]
"""
import json, os, random, re

from .eval import check
from .eval.practice import Cells
from .programs import lang
from .programs.python import compile_python
from .setup.log import log
from .setup.settings import settings, load_model, load_words
from .sudoku.boards import make_boards

FILES = ("program.py", "notes.md", "best.py")
WRITABLE = ("program.py", "notes.md")


def tools(p):
    """The tools, as described in config.yaml (prompts.agent.tools)."""
    return [{"type": "function", "function": {
        "name": name, "description": t["description"],
        "parameters": {"type": "object",
                       "properties": {k: {"type": v["type"], "description": v["description"]}
                                      for k, v in t.get("parameters", {}).items()},
                       "required": t.get("required", [])}}}
        for name, t in p["tools"].items()]


class Workspace:
    """The writer's files, in <out>/workspace."""

    def __init__(self, where):
        self.where = where
        os.makedirs(where, exist_ok=True)
        for f in FILES:
            open(self.path(f), "w").write("")

    def path(self, name):
        if name not in FILES:
            raise ValueError(f"there is no file {name!r}; the files are {', '.join(FILES)}")
        return os.path.join(self.where, name)

    def read(self, name):
        text = open(self.path(name)).read()
        return text if text else f"({name} is empty)"

    def write(self, name, content):
        if name not in WRITABLE:
            raise ValueError(f"{name} is kept by the harness; you can write {' and '.join(WRITABLE)}")
        open(self.path(name), "w").write(content)
        return f"wrote {name} ({len(content.splitlines())} lines)"


class Writer:
    """An MLX model that writes with thinking and native tool calls."""

    def __init__(self, path, a, tools):
        from mlx_lm import load
        from mlx_lm.sample_utils import make_sampler
        self.model, self.wrap = load(path)
        self.sampler = make_sampler(temp=a["temperature"], top_p=0.95, top_k=20)
        self.think, self.answer, self.tools = a["think_tokens"], a["answer_tokens"], tools

    def _generate(self, prompt, max_tokens, stop):
        from mlx_lm import stream_generate
        text = ""
        for r in stream_generate(self.model, self.wrap, prompt, max_tokens=max_tokens, sampler=self.sampler):
            text += r.text
            if stop in text:
                break
        return text

    def reply(self, chat):
        """Think for at most think_tokens; if the thinking is not done by then, close it and let the
        writer give its tool call."""
        prompt = self.wrap.apply_chat_template(chat, tools=self.tools, add_generation_prompt=True,
                                               enable_thinking=True)
        text = self._generate(prompt, self.think, "</think>")
        if "</think>" not in text:
            text += "\n</think>\n\n"
        prompt = prompt + self.wrap.encode(text, add_special_tokens=False)
        return text + self._generate(prompt, self.answer, "</tool_call>")


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
    ws = Workspace(f"{a.out}/workspace")
    best, best_right, done = "", plain, False
    history = []
    for call in range(1, ac["calls"] + 1):
        text = writer.reply(chat)
        thought, _, said = text.rpartition("</think>") if "</think>" in text else (text, "", "")
        calls = tool_calls(said)
        chat.append({"role": "assistant", "content": said.strip()})
        history.append(dict(call=call, thinking=thought.replace("<think>", "").strip(), reply=said.strip(), results=[]))
        if not calls:
            chat.append({"role": "user", "content": p["no_tool"]})
            log(f"call {call}: no tool call")
            json.dump(dict(system=system, log=history), open(f"{a.out}/session.json", "w"), indent=1)
            continue
        for name, args in calls:
            try:
                if name == "read_file":
                    result = ws.read(args["path"])
                    log(f"call {call}: read_file {args['path']}")
                elif name == "write_file":
                    result = ws.write(args["path"], args["content"])
                    log(f"call {call}: write_file {args['path']}\n    " + args["content"].replace("\n", "\n    "))
                elif name == "run":
                    code = args.get("code") or ""
                    source = code if code.strip() else open(ws.path("program.py")).read()
                    result, right = cells.run(source, int(args.get("first", 0)), int(args.get("count", 8)))
                    if right > best_right:
                        best, best_right = source, right
                        open(ws.path("best.py"), "w").write(best)
                        result += f"\nNew best: {right} cells right. This code is saved as best.py."
                    else:
                        result += f"\nBest so far: {best_right} cells right (best.py)."
                    log(f"call {call}: run {'code' if code.strip() else 'program.py'}, {right} right (best {best_right})\n    "
                        + source.replace("\n", "\n    "))
                elif name == "submit":
                    done, result = True, "ended; best.py will be checked"
                    log(f"call {call}: submit")
                else:
                    result = args if name == "error" else f"there is no tool {name}"
            except (ValueError, KeyError, IndexError, TypeError) as e:
                result = f"error: {e}"
                log(f"call {call}: {name}, error: {e}")
            chat.append({"role": "tool", "content": result})
            history[-1]["results"].append(result)
        json.dump(dict(system=system, log=history), open(f"{a.out}/session.json", "w"), indent=1)
        if done:
            break

    if not best:
        log(f"no program beat no program ({plain} of {len(cells.cells)} practice cells right)")
        return
    open(f"{a.out}/best.py", "w").write(best)
    log(f"best program, {best_right} of {len(cells.cells)} practice cells right:\n" + best)
    log("check:\n" + check.text(check.check(m, compile_python(best), practice_boards, test_boards)))


if __name__ == "__main__":
    main()
