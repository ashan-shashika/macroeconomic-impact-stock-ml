from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR  = Path("../results")
OUTPUT_DIR.mkdir(exist_ok=True)


def save(name: str, subpath: str):
    "Saves the current matplotlib figure to the results/*** directory with a given name."
    path = OUTPUT_DIR / subpath
    path.mkdir(parents=True, exist_ok=True)

    file = path / f"{name}.png"
    plt.savefig(file, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {file}")
