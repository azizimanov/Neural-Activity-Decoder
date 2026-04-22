import numpy as np
from project_brain_decoder.config import get_project_root
from project_brain_decoder.models.kalman import KalmanDecoder
from results import save_r2

np.random.seed(42)

# Load and split files
folder = get_project_root() / "data" / "raw"
files = sorted(folder.glob("*.nwb"))
train = files[:187]
val = files[187:249]
test = files[249:]

# Train
decoder = KalmanDecoder(n_components=50, dim_state=5)
decoder.fit(train)
