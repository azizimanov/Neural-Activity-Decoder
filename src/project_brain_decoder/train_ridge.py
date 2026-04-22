import numpy as np
from project_brain_decoder.config import get_project_root
from project_brain_decoder.models.ridge import RidgeDecoder
from results import save_r2

np.random.seed(42)

# Load and split files
folder = get_project_root() / "data" / "raw"
files = list(folder.glob("*.nwb"))
train = files[:187]
val = files[187:249]
test = files[249:]


# Train
decoder = RidgeDecoder(alpha=1.0)
decoder.fit(train)

# Evaluate on test sessions
mean_r2 = decoder.evaluate(test)
print(f"Index vel. R²: {mean_r2[0]:.4f}. MRS vel. R²: {mean_r2[1]:.4f}")


if __name__ == "__main__":
    main(path=data_dir, model=Ridge(), results_dir=results_dir)