"""Execute the experiment notebook in place (restart + run all)."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "notebooks" / "Spam_Detection_Experiments.ipynb"
nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=None, kernel_name="python3",
               resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)
errors = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
print("executed", path, "| errors:", len(errors))
