"""Basic questions about an agent session (runs/<name>/session.json), one line per call:
did the writer make a tool call, which tool, does its code parse as Python, how long it
thought, and the first line of what came back.

    python -m telepathy.report runs/agent_basic
"""
import ast, json, sys


def parses(code):
    try:
        ast.parse(code)
        return f"{len(code.splitlines())} lines, parses"
    except SyntaxError as e:
        return f"syntax error line {e.lineno}: {e.msg}"


def main(path):
    calls = json.load(open(f"{path}/session.json"))["log"]
    print(f"{len(calls)} calls; {sum('<tool_call>' in c['reply'] for c in calls)} with a tool call")
    for c in calls:
        tools = []
        for body in c["reply"].split("<tool_call>")[1:]:
            try:
                t = json.loads(body.split("</tool_call>")[0])
                args = t.get("arguments", {})
                code = args.get("content") if t["name"] == "write_file" and args.get("path") == "program.py" else args.get("code")
                tools.append(t["name"] + (f" {args['path']}" if "path" in args else "") + (f" ({parses(code)})" if code else ""))
            except (json.JSONDecodeError, KeyError) as e:
                tools.append(f"unreadable tool call: {e}")
        result = (c["results"][0].splitlines()[0] if c["results"] else "")[:80]
        print(f"call {c['call']}: thought {len(c['thinking'])} chars; {', '.join(tools) or 'no tool'}; result: {result}")


if __name__ == "__main__":
    main(sys.argv[1])
