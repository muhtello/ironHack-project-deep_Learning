
import json
import sys


def convert(py_path, ipynb_path):
    with open(py_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    # Group lines into raw blocks, splitting whenever a line starts a new
    # '# %%' marker (with or without a '[markdown]' tag).
    blocks = []
    current = []
    for line in lines:
        if line.startswith("# %%"):
            if current:
                blocks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        blocks.append(current)

    cells = []
    for block in blocks:
        header, body_lines = block[0], block[1:]
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)
        while body_lines and not body_lines[-1].strip():
            body_lines.pop()
        if not body_lines:
            continue

        if "[markdown]" in header:
            text_lines = [
                line[2:] if line.startswith("# ") else line.lstrip("#")
                for line in body_lines
            ]
            cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [line + "\n" for line in text_lines[:-1]] + [text_lines[-1]],
            }
        else:
            cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + "\n" for line in body_lines[:-1]] + [body_lines[-1]],
            }
        cells.append(cell)

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    with open(ipynb_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1)


if __name__ == "__main__":
    convert(sys.argv[1], sys.argv[2])
