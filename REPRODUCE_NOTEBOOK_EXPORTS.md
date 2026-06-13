# Reproduce Notebook Exports

Run from the repository root:

SOURCE_NOTEBOOK_DIR="$HOME/Downloads/ARCF-Net/github"
export SOURCE_NOTEBOOK_DIR
source .venv/bin/activate
python tools/export_notebooks_for_github.py
