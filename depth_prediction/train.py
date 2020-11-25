import os.path
import os
import torch
import random
import imageio

from depth_prediction.build_depth_dataset import load_depth_dataset
from depth_prediction.depth_prediction_net import create_depth_pred
from run_nerf_helpers import to8b

dirname = os.path.dirname(__file__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def loss_mse(pred, target):
    return torch.mean((pred - target) ** 2)


def evaluate(rays, depths, model, batch_size):
    length = rays.shape[0]

    rolling_sum = 0.0

    for i in range(0, length, batch_size):
        start = i
        end = min(length, i + batch_size)

        rays_batch = rays[start:end]
        depths_batch = depths[start:end]

        depths_pred = model(rays_batch)

        rolling_sum += torch.sum((depths_pred - depths_batch) ** 2).item()

    return rolling_sum / length


def save_depthmap(filename, depth, H, W):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    disp_map_label = 1 - 1. / torch.max(torch.ones_like(depth).cuda() * 1e-10,
                                        depth)
    disp_map_label = torch.reshape(disp_map_label, (H, W))

    rgb8_label = to8b(disp_map_label.cpu().detach().numpy())
    imageio.imwrite(filename, rgb8_label)


def train():
    learning_rate = 0.0005
    dataset_file = os.path.join(dirname, "depth_datasets/lego/train.pt")
    dataset_val_file = os.path.join(dirname, "depth_datasets/lego/val.pt")
    n_epochs = 200000
    batch_size = 65536
    print_epoch_n = 100
    H = 400
    W = 400

    model, optimiser = create_depth_pred(learning_rate)
    model.to(device)

    # TODO(mackopes): use trainloader
    rays_d_train, rays_o_train, depths_train = load_depth_dataset(dataset_file)
    rays_train = torch.cat([rays_d_train, rays_o_train], dim=1)

    rays_d_val, rays_o_val, depths_val = load_depth_dataset(dataset_val_file)
    rays_val = torch.cat([rays_d_val, rays_o_val], dim=1)

    data_l = rays_train.shape[0]

    for epoch in range(n_epochs):
        if epoch % print_epoch_n == 0:
            print(f"epoch {epoch}")

        batch_i = random.sample(range(data_l), batch_size)
        batch_rays = rays_train[batch_i]
        batch_depths = depths_train[batch_i]

        optimiser.zero_grad()

        pred_depths = model(batch_rays)
        loss = loss_mse(pred_depths, batch_depths)

        loss.backward()
        optimiser.step()

        if epoch % print_epoch_n == 0:
            print(f"Loss: {loss}")
            loss_val = evaluate(rays_val, depths_val, model, batch_size)
            print(f"Loss val: {loss_val}")

            filename_pred = os.path.join(dirname, f"logs/run/pred_{epoch}.jpg")
            filename_label = os.path.join(dirname, f"logs/run/label_{epoch}.jpg")

            rays_img = rays_train[:H * W]
            depths_img = depths_train[:H * W]
            save_depthmap(filename_label, depths_img, H, W)

            depths_img_pred = model(rays_img)
            save_depthmap(filename_pred, depths_img_pred, H, W)


if __name__ == "__main__":
    train()
