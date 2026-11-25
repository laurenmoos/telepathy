"""Basic questions about an agent session (runs/<name>/session.json), one line per call:
did the writer make a tool call, which tool, did the code compile, how many statements,
how long it thought, and what came back.

    python -m telepathy.report runs/agent_basic
"""
import json, sys

from .programs import lang


def compiles(code):
    """Does the code parse? (Word names are not checked: that needs the words loaded.)"""
    try:
        return len(lang.split_statements(code)) if code.strip() else 0, ""
    except ValueError as e:
        return None, str(e)


def main(path):
    s = json.load(open(f"{path}/session.json"))
    calls = s["log"]
    made = sum("<tool_call>" in c["reply"] for c in calls)
    print(f"{len(calls)} calls; {made} with a tool call")
    for c in calls:
        tool = "none"
        if "<tool_call>" in c["reply"]:
            try:
                body = json.loads(c["reply"].split("<tool_call>")[1].split("</tool_call>")[0])
                tool, code = body["name"], body.get("arguments", {}).get("code", "")
                n, err = compiles(code)
                tool += f", {n} statements" if n is not None else f", does not parse: {err}"
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                tool = f"unreadable tool call: {e}"
        result = (c["results"][0].splitlines()[0] if c["results"] else "")[:90]
        print(f"call {c['call']}: thought {len(c['thinking'])} chars; {tool}; result: {result}")


if __name__ == "__main__":
    main(sys.argv[1])
