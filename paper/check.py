"""Check main.tex for the errors a compiler would catch, since there is none.

This machine has no pdflatex, xelatex or latexmk (verified 2026-09-19), so
nothing here can claim the paper builds. What it CAN do is rule out the
mistakes that cost a compile cycle on Overleaf: a \cite key with no bibtex
entry, a \csSomething macro used but never defined, an unbalanced
environment, a missing figure file.

Run: python3 paper/check.py
"""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def read(name: str) -> str:
    return (HERE / name).read_text(encoding="utf-8")


def strip_comments(text: str) -> str:
    """Drop % comments, keeping \\% escapes. Commented-out code is not code."""
    return "\n".join(re.sub(r"(?<!\\)%.*$", "", line) for line in text.split("\n"))


def expand_inputs(text: str, depth: int = 0) -> str:
    """Splice in every \\input{...}, so a macro defined in another file counts.

    Without this the checker reported three macros as undefined the moment the
    number block moved into the generated numbers.tex -- a false alarm, and a
    checker that cries wolf gets ignored.
    """
    if depth > 4:
        return text
    def splice(match: "re.Match") -> str:
        name = match.group(1)
        for candidate in (HERE / name, HERE / f"{name}.tex"):
            if candidate.exists():
                return expand_inputs(candidate.read_text(encoding="utf-8"),
                                     depth + 1)
        return match.group(0)
    return re.sub(r"\\input\{([^}]+)\}", splice, text)


def main() -> int:
    tex = expand_inputs(read("main.tex"))
    live = strip_comments(tex)
    bib = read("refs.bib")
    problems, notes = [], []

    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    cited = {key.strip()
             for group in re.findall(r"\\cite\{([^}]*)\}", live)
             for key in group.split(",")}
    for key in sorted(cited - keys):
        problems.append(f"\\cite{{{key}}} has no entry in refs.bib")
    # A cite that only ever appears in a comment is a plan, not an error.
    planned = {key.strip()
               for group in re.findall(r"\\cite\{([^}]*)\}", tex)
               for key in group.split(",")} - cited
    if planned:
        notes.append(f"{len(planned)} key(s) cited only inside comments: "
                     + ", ".join(sorted(planned)))

    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", tex))
    used = set(re.findall(r"\\(\w+)", live))

    def occurrences(name: str, text_: str) -> int:
        """Uses of \name, not counting a longer macro that starts the same."""
        return len(re.findall(r"\\" + name + r"(?![A-Za-z])", text_))

    # The \newcommand line is itself live text, so a macro that is only
    # defined still occurs once. Anything above one is a real use. Getting
    # this wrong made the check vacuous on its first run.
    numbers = sorted(n for n in defined
                     if re.fullmatch(r"(cs|ref|spec|tx)[A-Z]\w*", n))
    idle = [n for n in numbers if occurrences(n, live) < 2]
    if idle:
        notes.append(f"{len(idle)} number(s) defined but not yet used in live "
                     f"text (they are still only in comments): "
                     + ", ".join("\\" + n for n in idle))
    for name in sorted(used):
        if re.fullmatch(r"(cs|ref|spec|tx)[A-Z]\w*", name) and name not in defined:
            problems.append(f"\\{name} is used but never defined")

    opened = re.findall(r"\\begin\{(\w+\*?)\}", live)
    closed = re.findall(r"\\end\{(\w+\*?)\}", live)
    for kind in sorted(set(opened) | set(closed)):
        if opened.count(kind) != closed.count(kind):
            problems.append(f"environment {kind}: {opened.count(kind)} begin, "
                            f"{closed.count(kind)} end")

    if live.count("{") != live.count("}"):
        problems.append(f"braces unbalanced: {live.count('{')} open, "
                        f"{live.count('}')} close")

    for graphic in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", live):
        candidates = [HERE / graphic] + [HERE / f"{graphic}{ext}"
                                         for ext in (".png", ".pdf", ".jpg")]
        if not any(path.exists() for path in candidates):
            problems.append(f"figure not found: {graphic}")

    for required in ("spconf.sty", "IEEEbib.bst"):
        if not (HERE / required).exists():
            problems.append(f"missing style file: {required}")

    todos = len(re.findall(r"\\TODO\{", tex))
    print(f"cites resolved      {len(cited & keys)}/{len(cited)}")
    print(f"numbers defined     {len(defined)}")
    print(f"TODO markers left   {todos}")
    for note in notes:
        print(f"  note: {note}")
    for problem in problems:
        print(f"  PROBLEM: {problem}")
    print("no structural problems found" if not problems
          else f"{len(problems)} problem(s) -- fix before uploading")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
