# %% [markdown]
# # Step 1 — Fix and verify Animals10 label mapping
#
# The original `translate.py` mixed Italian->English and English->Italian
# training. This exposes `build_labeled_dataset()` so `main.ipynb` can call


# %%
import os
import pandas as pd

# %%
def find_project_root(marker="archive.zip"):
    # Jupyter runs notebooks with the notebook's own folder as cwd, while a
    # plain `python script.py` run typically has the project root as cwd.
    # Walk upward from cwd until we find a file that only exists at the root,
    # so this works the same either way.
    path = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(path, marker)):
        parent = os.path.dirname(path)
        if parent == path:
            raise FileNotFoundError(f"Could not locate project root (marker: {marker})")
        path = parent
    return path

# %%
# One direction only: Italian folder name -> English label.
IT_TO_EN = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

# %%
def build_labeled_dataset(project_root=None, save_csv=True):
    """Verify the IT_TO_EN mapping against raw-img/, rebuild the image index
    with project-root-relative paths, and (optionally) save
    animals10_labeled.csv. Returns the DataFrame."""
    root = project_root or find_project_root()
    raw_img_dir = os.path.join(root, "extracted_animals", "raw-img")

    actual_folders = {
        name
        for name in os.listdir(raw_img_dir)
        if os.path.isdir(os.path.join(raw_img_dir, name))
    }
    mapped_folders = set(IT_TO_EN.keys())

    missing_from_dict = actual_folders - mapped_folders
    extra_in_dict = mapped_folders - actual_folders
    if missing_from_dict:
        raise ValueError(f"Folders on disk with no mapping: {missing_from_dict}")
    if extra_in_dict:
        raise ValueError(f"Mapping entries with no matching folder: {extra_in_dict}")

    records = []
    for it_label in sorted(actual_folders):
        folder = os.path.join(raw_img_dir, it_label)
        for filename in os.listdir(folder):
            full_path = os.path.join(folder, filename)
            # Relative + forward slashes so the CSV works on any machine/OS,
            # not just this one at this drive path.
            relative_path = os.path.relpath(full_path, root).replace(os.sep, "/")
            records.append(
                {
                    "image_path": relative_path,
                    "label_it": it_label,
                    "label_en": IT_TO_EN[it_label],
                }
            )

    df = pd.DataFrame(records)

    if save_csv:
        df.to_csv(os.path.join(root, "animals10_labeled.csv"), index=False)

    return df

# %%
if __name__ == "__main__":
    dataset = build_labeled_dataset()
    print(f"Verified {dataset['label_it'].nunique()} classes, mapping is 1:1 with folders on disk.")
    print(f"Total images indexed: {len(dataset)}")
    print(dataset["label_en"].value_counts())
