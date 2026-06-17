import utils
from dataset import SingleShapeDataset
from utils import plot_save_output
import torch
import torch.utils.data
from pathlib import Path


dataset_test = SingleShapeDataset(10)
data_loader_test = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False, num_workers=0,
    collate_fn=utils.collate_fn)
 

num_classes = 4
 
# get the model using the helper function
model = utils.get_instance_segmentation_model(num_classes)
# Use a low score threshold so short CPU training runs still produce
# qualitative predictions for inspection.
model.roi_heads.score_thresh = 0.0
results_dir = Path("results")
existing_checkpoints = sorted(results_dir.glob("maskrcnn_*.pth"))
if not existing_checkpoints:
    raise FileNotFoundError(
        "No checkpoint found in results/. Run train.py first so visualize.py can load a model."
    )

# checkpoint_path = existing_checkpoints[-1]
checkpoint_path = "results/maskrcnn_4.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
print(f"Loaded checkpoint: {checkpoint_path}")


model.eval()
path = "results/" 
for i in range(4):
    imgs, labels = dataset_test[i]
    output = model([imgs])
    plot_save_output(path+str(i)+"_result.png", imgs, output[0])
