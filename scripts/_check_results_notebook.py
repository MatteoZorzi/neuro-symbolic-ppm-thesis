"""Oracolo: esegue il notebook dei risultati e verifica che tutte le celle girino.

Il notebook e' il materiale della presentazione: se una cella non gira, si
scopre davanti ai prof. Qui viene eseguito end-to-end con lo stesso kernel del
progetto e vengono controllati gli output attesi.
"""

import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
notebook_path = ROOT / "notebooks" / "3_petrinets" / "Final_Matrix_Results.ipynb"

notebook = nbformat.read(notebook_path, as_version=4)
code_cells = [c for c in notebook.cells if c.cell_type == "code"]
print(f"[nb   ] {notebook_path.name}: {len(notebook.cells)} celle "
      f"({len(code_cells)} di codice, {len(notebook.cells) - len(code_cells)} di testo)")

client = NotebookClient(
    notebook, timeout=600, kernel_name="python3",
    resources={"metadata": {"path": str(notebook_path.parent)}},
)
client.execute()
print("[exec ] tutte le celle eseguite senza errori")

# --- controlli sugli output prodotti
figures = sum(
    "image/png" in output.get("data", {})
    for cell in notebook.cells for output in cell.get("outputs", [])
)
tables = sum(
    "text/html" in output.get("data", {})
    for cell in notebook.cells for output in cell.get("outputs", [])
)
text = "\n".join(
    output.get("text", "")
    for cell in notebook.cells for output in cell.get("outputs", [])
    if output.output_type == "stream"
)
print(f"[out  ] {figures} figure, {tables} tabelle renderizzate")
assert figures >= 4, "mancano delle figure"
assert tables >= 4, "mancano delle tabelle"
assert "900 run" in text, "il conteggio delle run non compare nell'output"

nbformat.write(notebook, notebook_path)
print(f"[save ] output salvati dentro {notebook_path.name}")
print("\nTUTTO VERDE")
