import json

with open("kaggle/kaggle_physsl_medmnist_benchmark.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Jupyter Notebook Validated Successfully!")
print(f"Format: v{nb['nbformat']}.{nb['nbformat_minor']}")
print(f"Total cells: {len(nb['cells'])}")
for i, c in enumerate(nb['cells']):
    first_line = c['source'][0].strip() if c['source'] else ''
    print(f"  Cell {i+1} [{c['cell_type']}]: {first_line[:50]}...")
