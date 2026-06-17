import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "RNN"))

from transformer_caption import TransformerCaptioner
from utils.coco_utils import decode_captions, load_coco_data


def sample_minibatch(data, batch_size):
    idx = np.random.choice(data["train_captions"].shape[0], batch_size)
    captions = data["train_captions"][idx]
    image_idxs = data["train_image_idxs"][idx]
    features = data["train_features"][image_idxs]
    return features, captions


def main():
    np.random.seed(233)
    torch.manual_seed(233)

    os.makedirs("results", exist_ok=True)
    small_data = load_coco_data(max_train=50)
    model = TransformerCaptioner(
        word_to_idx=small_data["word_to_idx"],
        input_dim=small_data["train_features"].shape[1],
        hidden_dim=128,
        wordvec_dim=128,
        num_heads=4,
        num_layers=1,
        max_length=max(30, small_data["train_captions"].shape[1]),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    loss_history = []
    num_epochs = 100
    batch_size = 25
    iterations_per_epoch = max(small_data["train_captions"].shape[0] // batch_size, 1)

    model.train()
    for t in range(num_epochs * iterations_per_epoch):
        features, captions = sample_minibatch(small_data, batch_size)
        optimizer.zero_grad()
        loss = model.loss(features, captions)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())

        if t % 10 == 0:
            print(f"(Iteration {t + 1}) loss: {loss.item():.6f}")

    plt.plot(loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Transformer captioning loss history")
    plt.savefig("results/transformer_loss_history.png")
    plt.close()

    model.eval()
    for split in ["train", "val"]:
        for i in range(2):
            data_dict = np.load(f"../RNN/datasets/samples/{split}_{i}.npy", allow_pickle=True).item()
            feature = data_dict["feature"].reshape(1, -1)
            image = plt.imread(f"../RNN/datasets/samples/{split}_{i}.png")

            sample_captions = model.sample(feature, max_length=30)
            sample_captions = decode_captions(sample_captions, small_data["idx_to_word"])

            plt.figure(figsize=(8, 4))
            plt.imshow(image)
            plt.title(
                "Your prediction: %s\nGT: %s"
                % (sample_captions[0], data_dict["gt_caption"])
            )
            plt.axis("off")
            plt.savefig(f"results/pred_{split}_{i}.png")
            plt.close()


if __name__ == "__main__":
    main()
