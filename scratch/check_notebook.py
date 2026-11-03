import json

with open("kaggle/kaggle_physsl_medmnist_benchmark.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    ctype = c['cell_type']
    lines = len(c['source'])
    chars = sum(len(l) for l in c['source'])
    first_line = c['source'][0].strip() if c['source'] else ""
    print(f"Cell {i+1} [{ctype}]: {lines} lines, {chars} chars | Header: {first_line[:60]}")

