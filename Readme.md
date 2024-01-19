# [WIP] ML Prediction: Trajectory Prediction 


## Table of Contents
- [Getting Started](#getting-started)
- [Directory Structure](#directory-structure)
- [To-Do](#to-do)


## Getting Started
To run the lstm-model (Work In Progress), run the following command:

```bash
pip3 install -r third_party/requirements.txt
python3 social_lstm/main.py
```


## Directory Structure
- `data/`: Directory for storing dataset files. Referenced from https://github.com/fjhheras/social-lstm-tf.
    - Raw data files are stored in directories `eth/` and `ucy`.
    - `preprocessed_data/`: Directory for storing preprocessed data files.
- [WIP] `social_lstm/`: Directory containing Python scripts for social-lstm model.
    - `src/`: Directory containing Python helper functions and classes for social-lstm model.
    - `main.py`: Main script for running social-lstm model.


## To-Do
The model is currently under devlopment.

- [ ] 1. Dataloader: train/val data split (currently only training data).
- [ ] 2. Dataloader: all datasets (currently only one dataset for debugging purpose).
- [ ] 3. Social-LSTM Model: implement occupancy grid and social embedding layer (currently using a placeholder replicating the input embedding layer).
- [ ] 4. Training: implement log likelyhood loss in closed-form (currently using MSE).
- [ ] 4. Envrionment: set up Bazil environment (having problem setting up tensorflow).