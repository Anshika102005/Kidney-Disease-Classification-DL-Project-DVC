import os, sys, json, zipfile
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
sys.path.insert(0, "D:\\Projects\\Kidney-Disease-Classification-DL-Project-DVC")

model_path = "D:\\Projects\\Kidney-Disease-Classification-DL-Project-DVC\\artifacts\\training\\trained_model.keras"

# Inspect the weights store structure
import h5py
from keras.saving.saving_lib import H5IOStore

with zipfile.ZipFile(model_path, "r") as z:
    # Read raw weights file
    raw = z.read("model.weights.h5")
    with open("_temp_weights.h5", "wb") as f:
        f.write(raw)

# Inspect H5 structure
with h5py.File("_temp_weights.h5", "r") as f:
    def print_group(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        else:
            print(f"  {name}/")
    
    print("Keys:", list(f.keys()))
    f.visititems(print_group)

os.unlink("_temp_weights.h5")
