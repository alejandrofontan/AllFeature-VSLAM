#!/usr/bin/env python3
"""Refresh the `#L<n>` line numbers of source links in the Markdown docs.

Convention (see docs/reference/README.md, "Links into the code"):

    [`track`](https://github.com/alejandrofontan/AllFeature-VSLAM/blob/main/src/Tracking.cc#L152)
    [`Tracking.cc`](https://github.com/.../src/Tracking.cc#L238 "else if (emergency_keyframe_)")
    [`# Main loop`](https://github.com/.../src/LoopClosing.cc#L73)

The line number is derived, never authoritative. What identifies the target is, in order:

1. the link *title* (the quoted string after the URL): a literal substring searched in the file;
2. a link text that is an identifier: the definition `<identifier>(` at column 0 (a function
   definition), else any `::<identifier>(`, else any `<identifier>` word;
3. a link text starting with `# `: the section banner `// # <name>`.

When several lines match, the one nearest to the current line number wins. Links whose text is
just the file name and that carry no title cannot be resolved and are reported. Relative targets
(`src/File.cc#L1`) are rewritten to the absolute GitHub form so the same page works on GitHub and
on the MkDocs site.

Usage:
    python docs/tools/resolve_links.py            # rewrite in place, print a report
    python docs/tools/resolve_links.py --check    # exit 1 if anything would change or is unresolved
    python docs/tools/resolve_links.py docs/reference/Tracking.md   # one file
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REPO_URL = "https://github.com/alejandrofontan/AllFeature-VSLAM/blob/main/"
# Pages that describe the code *as it is*: every link is refreshed.
DOC_GLOBS = ("docs/reference/*.md", "docs/index.md", "allfeature.md")
# Review pages: only the generated `## Checklist` table is refreshed; the dated notes below it are
# pinned to the commit they were written at and must keep their line numbers.
REVIEW_GLOB = "docs/review/*.md"
CHECKLIST_RE = re.compile(r"(?ms)^## Checklist\s*\n.*?(?=^## |\Z)")

LINK_RE = re.compile(
    r"\[(?P<text>[^\]]+)\]\("
    r"(?P<url>(?:" + re.escape(REPO_URL) + r")?(?P<path>(?:src|include|Thirdparty|settings)/[^\s#)\"]+|build\.sh|CMakeLists\.txt))"
    r"(?:#L(?P<line>\d+)(?:-L(?P<line2>\d+))?)?"
    r"(?:\s+\"(?P<title>[^\"]*)\")?"
    r"\)"
)

_file_cache: dict[pathlib.Path, list[str]] = {}


def lines_of(path: pathlib.Path) -> list[str] | None:
    if path not in _file_cache:
        try:
            _file_cache[path] = path.read_text(errors="replace").splitlines()
        except OSError:
            _file_cache[path] = None  # type: ignore[assignment]
    return _file_cache[path]


def nearest(candidates: list[int], current: int | None) -> int:
    if current is None or len(candidates) == 1:
        return candidates[0]
    return min(candidates, key=lambda n: abs(n - current))


def find_line(lines: list[str], text: str, title: str | None, current: int | None) -> int | None:
    text = text.strip("`")
    if title:
        hits = [i + 1 for i, l in enumerate(lines) if title in l]
        return nearest(hits, current) if hits else None
    if text.startswith("# "):
        banner = "// " + text
        hits = [i + 1 for i, l in enumerate(lines) if l.strip() == banner]
        return nearest(hits, current) if hits else None
    if re.fullmatch(r"[A-Za-z_~][\w:~]*", text):
        name = text.split("::")[-1]
        pat_def = re.compile(r"^\S.*\b" + re.escape(name) + r"\s*\(")   # definition at column 0
        pat_any = re.compile(r"::" + re.escape(name) + r"\s*\(")
        pat_word = re.compile(r"\b" + re.escape(name) + r"\b")
        for pat in (pat_def, pat_any, pat_word):
            hits = [i + 1 for i, l in enumerate(lines) if pat.search(l) and not l.lstrip().startswith("//")]
            if hits:
                return nearest(hits, current)
    return None


def process(doc: pathlib.Path, check: bool, checklist_only: bool = False) -> tuple[int, int, list[str]]:
    src = doc.read_text()
    changed = 0
    unresolved: list[str] = []
    skipped: list[str] = []

    def repl(m: re.Match) -> str:
        nonlocal changed
        text, path, line, line2, title = m["text"], m["path"], m["line"], m["line2"], m["title"]
        text_is_file_line = re.fullmatch(re.escape(pathlib.Path(path).name) + r"#L\d+", text.strip("`")) is not None
        target = REPO_ROOT / path
        lines = lines_of(target)
        if lines is None:
            # a target inside a submodule that is not checked out (CI clones without submodules):
            # nothing to verify, keep the link as written
            sub = REPO_ROOT / pathlib.Path(path).parts[0] / pathlib.Path(path).parts[1] if len(pathlib.Path(path).parts) > 2 else None
            if path.startswith("Thirdparty/") and sub is not None and sub.is_dir() and not any(sub.iterdir()):
                skipped.append(f"{doc}: {path} (submodule not checked out)")
                return m.group(0)
            unresolved.append(f"{doc}: {path} does not exist ({text})")
            return m.group(0)
        if line2 is not None:  # ranges are kept as written
            new_url = REPO_URL + path
            out = f"[{text}]({new_url}#L{line}-L{line2}" + (f' "{title}"' if title else "") + ")"
            changed += out != m.group(0)
            return out
        current = int(line) if line else None
        if current is None and not title:  # whole-file link: nothing to refresh, only normalise the URL
            out = f"[{text}]({REPO_URL}{path})"
            changed += out != m.group(0)
            return out
        found = find_line(lines, text, title, current)
        if found is None:
            if (text.strip("`") == pathlib.Path(path).name or text_is_file_line) and not title:
                unresolved.append(f"{doc}: [{text}]({path}#L{line}) — file-name link without a title pattern")
            else:
                unresolved.append(f"{doc}: [{text}]({path}#L{line}) — pattern not found")
            found = current
        if text_is_file_line and found:  # `File.cc#L12` link text carries the line too
            text = re.sub(r"#L\d+", f"#L{found}", text)
        out = f"[{text}]({REPO_URL}{path}" + (f"#L{found}" if found else "") + (f' "{title}"' if title else "") + ")"
        changed += out != m.group(0)
        return out

    if checklist_only:
        m = CHECKLIST_RE.search(src)
        if not m:
            return 0, 0, []
        new = src[:m.start()] + LINK_RE.sub(repl, m.group(0)) + src[m.end():]
        n_links = len(LINK_RE.findall(m.group(0)))
    else:
        new = LINK_RE.sub(repl, src)
        n_links = len(LINK_RE.findall(src))
    if new != src and not check:
        doc.write_text(new)
    for s in skipped:
        print("  skipped:", s)
    return changed, n_links, unresolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="Markdown files (default: all docs)")
    ap.add_argument("--check", action="store_true", help="report only; exit 1 if anything would change or is unresolved")
    args = ap.parse_args()

    if args.files:
        docs = [pathlib.Path(f).resolve() for f in args.files]
    else:
        docs = sorted({p for g in (*DOC_GLOBS, REVIEW_GLOB) for p in REPO_ROOT.glob(g)})
    docs = [d for d in docs if d.name != "README.md"]
    total_changed = total_links = 0
    all_unresolved: list[str] = []
    for doc in docs:
        checklist_only = doc.parent == REPO_ROOT / "docs" / "review"
        changed, n, unresolved = process(doc, args.check, checklist_only)
        total_changed += changed
        total_links += n
        all_unresolved += unresolved
        if changed or unresolved:
            rel = doc.relative_to(REPO_ROOT) if doc.is_absolute() else doc
            print(f"{rel}: {n} links, {changed} {'would change' if args.check else 'updated'}, {len(unresolved)} unresolved")
    for u in all_unresolved:
        print("  unresolved:", u)
    print(f"total: {total_links} links, {total_changed} {'would change' if args.check else 'updated'}, {len(all_unresolved)} unresolved")
    return 1 if args.check and (total_changed or all_unresolved) else 0


if __name__ == "__main__":
    sys.exit(main())
