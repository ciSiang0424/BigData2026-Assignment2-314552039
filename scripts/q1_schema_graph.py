"""Q1: draw the schema graph of the 8 Assignment-2 tables.

The graph is derived strictly from assignment2/SCHEMA.md (every row whose `key` cell is
"PK" or "FK → <Table>"), and is cross-checked against the TABLES dict in
assignment2/scripts/preprocess.py (the code that actually writes the tables).

Nodes  = the 8 tables (PK, FK columns and the row-time column are listed in each node).
Edges  = one per foreign key, drawn FK column -> referenced table's PK, labelled with the
         FK column name. FKs marked "known only later" in SCHEMA.md are dashed.

Usage:
    python q1_schema_graph.py [--repo ../assignment2] [--out .] [--dot /path/to/dot] [--all-columns]
    python q1_schema_graph.py --slide [--period "2025-12-01 00:00 - 2026-11-20 23:59 UTC"]
Outputs: schema_graph.dot / .png / .pdf / .svg  (schema_graph_full.* with --all-columns,
         schema_graph_slide.* with --slide: 16:9, left-to-right by table level, edges coloured by
         the referenced table, no edge labels -- each edge starts at its FK row, so the FK is still
         identified -- and an optional caption with the dataset period)
"""
import argparse
import ast
import re
import shutil
import subprocess
from pathlib import Path

TABLE_ORDER = ["Users", "Repositories", "PullRequests", "Issues",
               "IssueComments", "PRReviews", "PRReviewComments", "Pushes"]
FK_RE = re.compile(r"FK\s*(?:→|->)\s*(\w+)")


def parse_schema_md(path):
    """{table: [ {col, type, key, desc} ]} from the markdown tables of SCHEMA.md."""
    tables, cur = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^##\s+(\w+)\s*$", line)
        if m:
            cur = m.group(1)
            tables[cur] = []
            continue
        if cur and line.startswith("| `"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # description may contain escaped pipes (\|); only the first 3 cells matter
            col, typ, key = cells[0].strip("`"), cells[1], cells[2]
            tables[cur].append({"col": col, "type": typ, "key": key,
                                "desc": "|".join(cells[3:])})
    return tables


def schema_keys(tables):
    pks, fks, rowtime = {}, [], {}
    for t, cols in tables.items():
        for c in cols:
            if c["key"] == "PK":
                pks[t] = c["col"]
            m = FK_RE.search(c["key"])
            if m:
                fks.append((t, c["col"], m.group(1), "known only later" in c["key"]))
            if "row time" in c["key"]:
                rowtime[t] = c["col"]
    return pks, fks, rowtime


def preprocess_fks(path):
    """FK set from the TABLES literal in preprocess.py (parsed with ast, no imports)."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TABLES":
            spec = ast.literal_eval(node.value)
            return ({(t, col, ref) for t, s in spec.items() for col, ref in s["fks"].items()},
                    {t: s["pk"] for t, s in spec.items()},
                    {t: s["time"] for t, s in spec.items()})
    raise SystemExit("TABLES not found in preprocess.py")


def html_node(table, cols, pk, fk_cols, rowtime, all_columns):
    rows = [f'<TR><TD BGCOLOR="#1f4e79" COLSPAN="2"><FONT COLOR="white"><B>{table}</B></FONT></TD></TR>']
    for c in cols:
        name = c["col"]
        if name == pk:
            tag, bg = "PK", "#ffe699"
        elif name in fk_cols:
            tag, bg = "FK", "#c6e0b4"
        elif name == rowtime:
            tag, bg = "time", "#ddebf7"
        elif all_columns:
            tag, bg = "", "white"
        else:
            continue
        label = f"<B>{name}</B>" if tag == "PK" else (f"<I>{name}</I>" if tag == "FK" else name)
        rows.append(f'<TR><TD ALIGN="LEFT" BGCOLOR="{bg}" PORT="{name}_l">{label}</TD>' +
                    (f'<TD ALIGN="LEFT" BGCOLOR="{bg}" PORT="{name}_r"><FONT POINT-SIZE="9">{tag}</FONT></TD></TR>' if tag
                     else f'<TD BGCOLOR="{bg}" PORT="{name}_r"> </TD></TR>'))
    return f'<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">{"".join(rows)}</TABLE>>'


def build_dot(tables, pks, fks, rowtime, all_columns):
    out = ['digraph schema {', '  graph [rankdir=LR, splines=true, nodesep=0.5, ranksep=1.3, '
           'fontname="Helvetica", label="Assignment 2 schema graph: 8 tables, '
           f'{len(fks)} foreign keys (FK \u2192 PK).  PK = primary key, FK = foreign key, '
           'time = row-time column; dashed = FK known only later", labelloc=t, fontsize=16];',
           '  node [shape=plaintext, fontname="Helvetica", fontsize=11];',
           '  edge [fontname="Helvetica", fontsize=9, color="#444444", arrowsize=0.8];']
    for t in TABLE_ORDER:
        fk_cols = {c for (tt, c, _, _) in fks if tt == t}
        out.append(f'  {t} [label={html_node(t, tables[t], pks[t], fk_cols, rowtime.get(t), all_columns)}];')
    for t, col, ref, later in fks:
        style = ', style=dashed' if later else ''
        if t == ref:   # self reference (PRReviewComments.in_reply_to_id): loop on the right side
            out.append(f'  {t}:{col}_r:e -> {ref}:{pks[ref]}_r:e [label="{col}"{style}];')
        else:
            out.append(f'  {t}:{col}_r:e -> {ref}:{pks[ref]}_l:w [label="{col}"{style}];')
    out.append('}')
    return "\n".join(out)


SLIDE_RANKS = [["PRReviewComments", "IssueComments", "Pushes"], ["PRReviews", "Issues"],
               ["PullRequests"], ["Users", "Repositories"]]
EDGE_COLOR = {"Users": "#1f77b4", "Repositories": "#d62728", "PullRequests": "#2ca02c",
              "Issues": "#9467bd", "PRReviews": "#ff7f0e", "PRReviewComments": "#8c564b"}


def build_slide_dot(tables, pks, fks, rowtime, period):
    cap = ("Schema graph: 8 tables (nodes), 20 foreign keys (edges, FK \u2192 referenced PK). "
           "PK = primary key, FK = foreign key, time = row-time column; dashed = FK known only later (merged_by_id).")
    if period:
        cap += f"\\nDataset period used: {period}"
    out = ['digraph schema_slide {',
           f'  graph [rankdir=LR, splines=spline, nodesep=0.35, ranksep=1.6, size="13.33,7.5", ratio=fill, '
           f'fontname="Helvetica", label="{cap}", labelloc=b, fontsize=15];'.replace("\\u2192", "\u2192"),
           '  node [shape=plaintext, fontname="Helvetica", fontsize=14];',
           '  edge [penwidth=1.6, arrowsize=0.9];']
    for rank in SLIDE_RANKS:
        out.append("  { rank=same; " + " ".join(rank) + "; }")
    for t in TABLE_ORDER:
        fk_cols = {c for (tt, c, _, _) in fks if tt == t}
        out.append(f'  {t} [label={html_node(t, tables[t], pks[t], fk_cols, rowtime.get(t), False)}];')
    for t, col, ref, later in fks:
        style = ', style=dashed' if later else ''
        color = EDGE_COLOR[ref]
        if t == ref:
            out.append(f'  {t}:{col}_r:e -> {ref}:{pks[ref]}_r:e [color="{color}"{style}];')
        else:
            out.append(f'  {t}:{col}_r:e -> {ref}:{pks[ref]}_l:w [color="{color}"{style}];')
    out.append('}')
    return "\n".join(out)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(here.parent / "assignment2"))
    ap.add_argument("--out", default=str(here))
    ap.add_argument("--dot", default=shutil.which("dot") or str(here / "mamba/envs/gv/bin/dot"))
    ap.add_argument("--all-columns", action="store_true", help="also list non-key columns")
    ap.add_argument("--slide", action="store_true", help="slide-ready 16:9 version (schema_graph_slide.*)")
    ap.add_argument("--period", default="", help="caption for --slide, e.g. the exact UTC data period")
    a = ap.parse_args()

    repo, out = Path(a.repo), Path(a.out)
    tables = parse_schema_md(repo / "SCHEMA.md")
    assert sorted(tables) == sorted(TABLE_ORDER), sorted(tables)
    pks, fks, rowtime = schema_keys(tables)

    # Cross-check with the code that writes the data.
    code_fks, code_pks, code_time = preprocess_fks(repo / "scripts" / "preprocess.py")
    md_fks = {(t, c, r) for t, c, r, _ in fks}
    assert md_fks == code_fks, f"SCHEMA.md vs preprocess.py FK mismatch: {md_fks ^ code_fks}"
    assert pks == code_pks and rowtime == code_time, "PK / row-time mismatch"
    print(f"{len(tables)} tables, {len(fks)} FK edges (SCHEMA.md == preprocess.py):")
    for t, c, r, later in fks:
        print(f"  {t}.{c} -> {r}.{pks[r]}" + ("   (known only later)" if later else ""))

    assert len(fks) == 20, len(fks)
    if a.slide:
        suffix, dot_src = "_slide", build_slide_dot(tables, pks, fks, rowtime, a.period)
    else:
        suffix = "_full" if a.all_columns else ""
        dot_src = build_dot(tables, pks, fks, rowtime, a.all_columns)
    dot_path = out / f"schema_graph{suffix}.dot"
    dot_path.write_text(dot_src, encoding="utf-8")
    for fmt, extra in (("png", ["-Gdpi=200"]), ("pdf", []), ("svg", [])):
        subprocess.run([a.dot, f"-T{fmt}", *extra, str(dot_path), "-o",
                        str(out / f"schema_graph{suffix}.{fmt}")], check=True)
    print("wrote", ", ".join(str(out / f"schema_graph{suffix}.{x}") for x in ("dot", "png", "pdf", "svg")))


if __name__ == "__main__":
    main()
