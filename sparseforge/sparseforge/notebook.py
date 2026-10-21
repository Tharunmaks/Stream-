"""Generate a Kaggle/Colab notebook that runs the same config on a GPU (not executed remotely)."""
import json

def make_notebook(config, target="kaggle", repo_url=None):
    install = f"!pip install -q git+{repo_url}" if repo_url else "# upload the sparseforge folder, then:\n%cd sparseforge"
    code = f"""import json, torch
from sparseforge.train import run
cfg = json.loads('''{json.dumps(config)}''')
print('cuda:', torch.cuda.is_available())
# NOTE: sparseforge.train currently runs on CPU; on GPU use cfg as a reference for a CUDA trainer.
print(run(cfg))"""
    cells = [{"cell_type": "markdown", "metadata": {}, "source": [f"# SparseForge job ({target})\nEnable GPU in notebook settings."]},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [install]},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [code]}]
    return {"nbformat": 4, "nbformat_minor": 5, "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}}, "cells": cells}
