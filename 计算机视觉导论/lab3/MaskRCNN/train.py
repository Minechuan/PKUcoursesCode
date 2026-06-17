
import utils
from engine import train_one_epoch
from dataset import MultiShapeDataset
import torch
import torch.utils.data
import os
import time
from pathlib import Path

os.makedirs("results", exist_ok=True)
writer = utils.log_writer("results", "maskrcnn")
num_classes = 4 # 0 for backgroud 
 
model = utils.get_instance_segmentation_model(num_classes).double()
results_dir = Path("results")
existing_checkpoints = sorted(results_dir.glob("maskrcnn_*.pth"))
if existing_checkpoints:
    checkpoint_path = existing_checkpoints[-1]
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    print(f"Loaded checkpoint: {checkpoint_path}")
else:
    print("No existing checkpoint found in results/. Training from scratch.")

dataset = MultiShapeDataset(512)

torch.manual_seed(233)

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=8, num_workers=0, shuffle=True,
    collate_fn=utils.collate_fn)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.002,
                            momentum=0.9, weight_decay=0.0005)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.5)

num_epochs = 10
device = torch.device('cpu')

count = 0
for epoch in range(num_epochs):
    t0 = time.time()
    count = train_one_epoch(model, optimizer, data_loader, device, count, writer)
    torch.save(model.state_dict(), "results/maskrcnn_"+str(epoch)+".pth") 
    lr_scheduler.step()

    print(f"Epoch {epoch} finished, time: {int(time.time()-t0) / 60.0} min.")
