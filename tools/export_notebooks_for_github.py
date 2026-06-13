from pathlib import Path
import os
import re
import shutil
import subprocess
import sys
import nbformat

ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK_DIR = Path(os.environ.get("SOURCE_NOTEBOOK_DIR", str(Path.home() / "Downloads" / "ARCF-Net" / "github"))).expanduser()

SMALL_ORIGINAL_DIR = ROOT / "notebooks" / "original_small"
CLEAN_NOTEBOOK_DIR = ROOT / "notebooks" / "github_renderable"
MD_EXPORT_DIR = ROOT / "reports" / "notebook_exports"
PY_EXPORT_DIR = ROOT / "scripts" / "notebook_code_exports"

MAX_NORMAL_GIT_FILE_MB = 95
MARKDOWN_CHUNK_LIMIT_BYTES = 15 * 1024 * 1024

def fail(message):
    print("ERROR:", message, file=sys.stderr)
    sys.exit(1)

def safe_name(name):
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:130] or "notebook"

def size_mb(path):
    return path.stat().st_size / (1024 * 1024)

def has_outputs(nb):
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.get("outputs"):
            return True
    return False

def strip_outputs(nb):
    clean = nbformat.from_dict(nb)
    for key in ["widgets", "varInspector", "toc", "collapsed_sections"]:
        clean.metadata.pop(key, None)
    for cell in clean.cells:
        cell.metadata = {}
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return clean

def clean_keep_outputs(nb):
    cleaned = nbformat.from_dict(nb)
    for key in ["widgets", "varInspector", "toc", "collapsed_sections"]:
        cleaned.metadata.pop(key, None)
    for cell in cleaned.cells:
        cell.metadata.pop("execution", None)
        cell.metadata.pop("widgets", None)
    return cleaned

def write_python_export(nb, out_path, original_name):
    lines = [
        f"# Auto-exported from: {original_name}",
        "# Public Python export for GitHub visibility.",
        "# Notebook outputs are available in reports/notebook_exports/.",
        "",
    ]

    for i, cell in enumerate(nb.cells, start=1):
        if cell.cell_type == "markdown":
            lines.append(f"# %% [markdown] Cell {i}")
            for line in cell.source.splitlines():
                lines.append("# " + line)
            lines.append("")
        elif cell.cell_type == "code":
            lines.append(f"# %% Cell {i}")
            lines.append(cell.source.rstrip())
            lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")

def split_markdown(md_path):
    if not md_path.exists():
        return [], False

    if md_path.stat().st_size <= MARKDOWN_CHUNK_LIMIT_BYTES:
        return [md_path.name], False

    text = md_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)

    parts = []
    current = []
    current_size = 0
    part_no = 1

    def save_part(part_lines, number):
        part_name = f"{md_path.stem}_part_{number:02d}.md"
        part_path = md_path.parent / part_name
        header = f"# {md_path.stem} - Part {number:02d}\n\n[Back to Notebook Index](../../NOTEBOOK_INDEX.md)\n\n"
        part_path.write_text(header + "".join(part_lines), encoding="utf-8")
        return part_name

    for line in lines:
        line_size = len(line.encode("utf-8", errors="replace"))

        if current and current_size + line_size > MARKDOWN_CHUNK_LIMIT_BYTES:
            parts.append(save_part(current, part_no))
            part_no += 1
            current = []
            current_size = 0

        current.append(line)
        current_size += line_size

    if current:
        parts.append(save_part(current, part_no))

    md_path.unlink()
    return parts, True

def reset_folder(folder):
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True, exist_ok=True)

def main():
    print("Repository root:", ROOT)
    print("Source notebook folder:", SOURCE_NOTEBOOK_DIR)

    if not SOURCE_NOTEBOOK_DIR.exists():
        fail(f"Source folder not found: {SOURCE_NOTEBOOK_DIR}")

    notebooks = sorted(SOURCE_NOTEBOOK_DIR.rglob("*.ipynb"))

    if not notebooks:
        fail(f"No .ipynb files found in: {SOURCE_NOTEBOOK_DIR}")

    reset_folder(SMALL_ORIGINAL_DIR)
    reset_folder(CLEAN_NOTEBOOK_DIR)
    reset_folder(MD_EXPORT_DIR)
    reset_folder(PY_EXPORT_DIR)

    index_rows = []
    skipped_rows = []
    large_notes = []

    for idx, nb_path in enumerate(notebooks, start=1):
        original_name = nb_path.name
        original_size = size_mb(nb_path)
        stem = f"{idx:02d}_{safe_name(nb_path.stem)}"

        print("")
        print(f"Processing {idx}/{len(notebooks)}: {original_name}")
        print(f"Size: {original_size:.2f} MB")

        nb = nbformat.read(nb_path, as_version=4)
        output_status = "Yes" if has_outputs(nb) else "No"

        if original_size <= MAX_NORMAL_GIT_FILE_MB:
            raw_dest = SMALL_ORIGINAL_DIR / original_name
            shutil.copy2(nb_path, raw_dest)
            raw_link = f"[Small original](notebooks/original_small/{raw_dest.name})"
        else:
            raw_link = "Skipped because raw file is too large"
            skipped_rows.append(f"| `{original_name}` | {original_size:.2f} MB | Raw notebook skipped. Public exports generated. |")
            large_notes.append(f"- `{original_name}` is {original_size:.2f} MB, so raw notebook was not uploaded.")

        clean_nb = strip_outputs(nb)
        clean_path = CLEAN_NOTEBOOK_DIR / f"{stem}.ipynb"
        nbformat.write(clean_nb, clean_path)

        if size_mb(clean_path) > MAX_NORMAL_GIT_FILE_MB:
            clean_path.unlink(missing_ok=True)
            clean_link = "Skipped because clean notebook is still too large"
        else:
            clean_link = f"[Clean notebook](notebooks/github_renderable/{clean_path.name})"

        py_path = PY_EXPORT_DIR / f"{stem}.py"
        write_python_export(nb, py_path, original_name)

        output_nb = clean_keep_outputs(nb)
        temp_nb = MD_EXPORT_DIR / f"{stem}_temporary_with_outputs.ipynb"
        nbformat.write(output_nb, temp_nb)

        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "markdown",
                "--output",
                stem,
                "--output-dir",
                str(MD_EXPORT_DIR),
                str(temp_nb),
            ],
            check=True,
        )

        temp_nb.unlink(missing_ok=True)

        md_path = MD_EXPORT_DIR / f"{stem}.md"
        md_parts, was_split = split_markdown(md_path)

        if was_split:
            md_link = "<br>".join(
                f"[Part {part_no:02d}](reports/notebook_exports/{part})"
                for part_no, part in enumerate(md_parts, start=1)
            )
        elif md_parts:
            md_link = f"[Markdown output](reports/notebook_exports/{md_parts[0]})"
        else:
            md_link = "Markdown export failed"

        index_rows.append(
            f"| {idx} | `{original_name}` | "
            f"{raw_link} | "
            f"{clean_link} | "
            f"{md_link} | "
            f"[Python code](scripts/notebook_code_exports/{py_path.name}) | "
            f"{original_size:.2f} MB | {output_status} |"
        )

    index_md = [
        "# FedRL-FuseNet Notebook Visibility Index",
        "",
        "GitHub may fail to render large Jupyter notebooks directly.",
        "Use this page to view public notebook exports.",
        "",
        "## Best viewing order",
        "",
        "1. Open the Markdown output to see saved outputs.",
        "2. Open the Python code file to inspect all code.",
        "3. Open the clean notebook for lightweight viewing.",
        "4. Download the small original notebook only when available.",
        "",
        "Large raw notebooks are skipped if they are too large for normal GitHub upload.",
        "Their code and saved outputs are still exported.",
        "",
        "## Notebook files",
        "",
        "| No. | Source notebook | Raw original | Clean notebook | Markdown output | Python code | Source size | Has saved outputs |",
        "|---:|---|---|---|---|---|---:|---|",
        *index_rows,
        "",
    ]

    if large_notes:
        index_md += [
            "## Large raw notebook notes",
            "",
            *large_notes,
            "",
        ]

    (ROOT / "NOTEBOOK_INDEX.md").write_text("\n".join(index_md), encoding="utf-8")

    report = [
        "# Large File Report",
        "",
        "This report shows how large notebooks were handled.",
        "",
    ]

    if skipped_rows:
        report += [
            "## Raw notebooks skipped",
            "",
            "| Notebook | Size | Action |",
            "|---|---:|---|",
            *skipped_rows,
            "",
        ]
    else:
        report += [
            "No raw notebooks were skipped.",
            "",
        ]

    (ROOT / "LARGE_FILE_REPORT.md").write_text("\n".join(report), encoding="utf-8")

    readme = ROOT / "README.md"
    notice = "\n\n## Notebook visibility\n\nLarge Jupyter notebooks may not render directly on GitHub.\n\nFor public viewing of notebook code and outputs, open:\n\n[NOTEBOOK_INDEX.md](NOTEBOOK_INDEX.md)\n\nThis repository provides:\n- clean lightweight notebooks\n- Markdown exports with saved outputs\n- Python code exports\n- small original notebooks when safely uploadable\n"

    if readme.exists():
        old = readme.read_text(encoding="utf-8", errors="replace")
        if "NOTEBOOK_INDEX.md" not in old:
            readme.write_text(old.rstrip() + notice + "\n", encoding="utf-8")
    else:
        readme.write_text("# FedRL-FuseNet\n" + notice + "\n", encoding="utf-8")

    reproduce = ROOT / "REPRODUCE_NOTEBOOK_EXPORTS.md"
    reproduce.write_text(
        "# Reproduce Notebook Exports\n\n"
        "Run from the repository root:\n\n"
        "SOURCE_NOTEBOOK_DIR=\"$HOME/Downloads/ARCF-Net/github\"\n"
        "export SOURCE_NOTEBOOK_DIR\n"
        "source .venv/bin/activate\n"
        "python tools/export_notebooks_for_github.py\n",
        encoding="utf-8",
    )

    print("")
    print("Done.")
    print("Created NOTEBOOK_INDEX.md")
    print("Created public-visible Markdown, Python, and clean notebook exports")

if __name__ == "__main__":
    main()
