## Data
This project uses the LINK dataset. To download:

pip install dandi
dandi download -o data/raw DANDI:001201/0.251023.2336


1) LINK dataset from DANDI is already pre-binned at 20 ms intervals.
2) The dataset has already predefined LFP features.
3) The data in the extracted NWB arrays are already in a scaled down format (1e-06).
4) Neural and kinematics data are pre-aligned (matching timestamps).
5) Using a Linear Ridge as baseline.
6) Using Temporal CNN (TCN) for efficient long-range temporal modeling. Compared to GRU/LSTM, TCNs can capture long 
dependencies with parallel convolutions, making them faster to train and often more computationally efficient.