import numpy as np
from project_brain_decoder.config import get_project_root
from project_brain_decoder.models.ridge import RidgeDecoder
from results import save_r2

np.random.seed(42)

# Load and split files
folder = get_project_root() / "data" / "raw"
files = list(folder.glob("*.nwb"))
np.random.shuffle(files)
train = files[:187] # 60%
val = files[187:249] # 20%
test = files[249:] # 20%

# Train
decoder = RidgeDecoder(alpha=1.0)
decoder.fit(train)

# Evaluate on test sessions
mean_r2 = decoder.evaluate(test)
print(f"Index vel. R²: {mean_r2[0]:.2f}. MRS vel. R²: {mean_r2[1]:.2f}")

# Save model and scores
decoder.save(get_project_root() / "models" / "ridge.pkl")
save_r2.get_scores(model="ridge", score1=round(mean_r2[0], 2), score2=round(mean_r2[1], 2))