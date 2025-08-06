import scipy.io
import numpy as np
import os

# Pick a .mat that you know has obstacles (e.g. a later frame with buoys)
path = r"data\raw\modd\raw\MODD2_annotations_v2\annotations_v2_redone\kope67-00-00004500-00005050\ground_truth\00004550L.mat"
print("Inspecting:", path)

mat = scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
ann = mat.get('annotations')
if ann is None:
    raise ValueError("No 'annotations' key in mat")

# List the fields
fields = [f for f in dir(ann) if not f.startswith('_')]
print("Fields in annotations:", fields)

# Show sea_edge for sanity
print("\nSea edge (first 5):")
print(ann.sea_edge[:5])

# Inspect obstacles
obs = ann.obstacles
print("\nType of obstacles:", type(obs))
shape = getattr(obs, 'shape', None)
print("Shape of obstacles:", shape)

if isinstance(obs, np.ndarray):
    if obs.size == 0:
        print("→ obstacles is an empty array")
    else:
        print("First few rows of obstacles array:")
        print(obs[:min(5, obs.shape[0])])
elif isinstance(obs, (list, tuple)):
    print(f"List of length {len(obs)}; first element type:", type(obs[0]))
    print(obs[0])
else:
    print("Something else:", obs)
