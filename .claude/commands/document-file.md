---
name: document-file
description: Write or refresh the reference page of a source-file family in the docs/reference format. Usage: /document-file <Name> [<src_file> ...] — e.g. /document-file LocalMapping src/LocalMapping.cc src/LocalMapping_aux.cc include/LocalMapping.h
---

You are writing `docs/reference/<Name>.md`, the reference page of one source-file family, in the
format defined by `docs/reference/README.md`. The user gives the page name and, optionally, the
source files; when only the name is given, the family is `src/<Name>.cc`, `src/<Name>_aux.cc` (if
it exists) and `include/<Name>.h`.

The page says what the code does today. It carries no history and no measurements: those belong
to `docs/review/<Name>.md` and `docs/gym/`. Do not invent behaviour; every bullet must be
traceable to a line you read.

## Steps

### 1. Read the format and the sources
Read `docs/reference/README.md` in full. Read every source file of the family in full, header
included. Note the `// # banner` section comments of the `.cc`: they define the page's H2s. If the
file has none, you will group functions by call-graph order under H2s you choose (say so in the
page's opening paragraph).

### 2. List the functions and the call graph
List every function *defined* in the `.cc`/`.cpp` files (`Class::name(` at column 0). For each,
find its callers inside the family and outside it (`grep -rn "name(" src include`). Build the
`## Call graph` list: the thread body or public entry point first, then what it calls, then the
functions not in the graph (setup, reset, accessors).

### 3. Check the stock ORB-SLAM2 counterpart
For each function with an ORB-SLAM2 namesake, open the counterpart in `../ORB-SLAM2/src/` (the
reference copy next to this checkout; if it is missing, say so in your final message and skip this
step). Where the behaviour differs, note what and why; where it does not, note nothing.

### 4. Collect the settings keys
`grep -n 'read_if_present\|fSettings\["' <sources>`: every key read by the family, its compiled
default (from the header or the `LoadParameters` body) and the function that consumes it. These
fill the `## Settings read by this file` table and the `settings:` bullets.

### 5. Draft the page
Follow the skeleton in `docs/reference/README.md` exactly:
- H1 `# \`src/<Name>.cc\``, one opening paragraph (role, thread, what it owns);
- `## Call graph`;
- `## Flow` with one `flowchart TD` mermaid diagram, under ~15 nodes, palette `classDef` block,
  `click <node> "#<function>"` for every node that is a function of this file;
- one H2 per section banner (`## \`# Main loop\``) with the `**Section:**` link and an optional
  `flowchart LR`;
- one H3 per function, named exactly after the function, then the signature block copied from the
  definition, then bullets: contract, mechanism, `called from:`, `differs from ORB-SLAM2:` (only if
  it does), `settings:` (only if it reads any);
- `## Settings read by this file`.

Links into the code use the absolute form
`https://github.com/alejandrofontan/AllFeature-VSLAM/blob/main/src/<File>.cc#L<n>`. A link whose
text is not the target's identifier gets a title with a literal search pattern, e.g.
`[\`System.cc\`](…/src/System.cc#L211 "loop_closing_thread_ = ")`. Put the line number you read;
step 6 corrects it.

If a page already exists, keep its structure and every bullet that is still true; rewrite only
what the code no longer matches; do not delete function entries silently (say in your final message
which ones you removed and why).

### 6. Resolve the links
Run `python docs/tools/resolve_links.py docs/reference/<Name>.md`. Fix every `unresolved` report
(usually a file-name link that needs a title pattern). Run it again until it reports zero
unresolved.

### 7. Cross-check
- every function defined in the sources has an H3, and every H3 names a function that exists;
- every `click` target in the diagrams is an existing H3 anchor (`#<function>`);
- every settings key in the table is read by the sources, and every key read appears in the table;
- no bullet states a number, a date or a "fixed"/"was" — move those to the review page or a gym
  entry and link them.

### 8. Report
List: the functions documented, the ORB-SLAM2 differences found, the settings keys, anything you
could not determine from the code (and where you looked), and the resolver's final report. Do not
commit; the user reviews the page first.
