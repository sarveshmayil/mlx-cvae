import argparse
from datetime import datetime
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.optimizers import AdamW
from mlx.utils import tree_flatten
from PIL import Image
from tqdm import tqdm

from cvae import CVAE
from dataset import mnist


def reconstruct(model: CVAE, batch: dict[str, mx.array], save_path: Path):
    images = mx.array(batch["image"])
    x_recon, _, _ = model(images)  # (B, H, W, C)

    # (B, H, W, C) -> (2, B, H, W, C) -> (B, 2, H, W, C) -> (B * 2, H, W, C)
    pairs = mx.stack([images, x_recon]).swapaxes(0, 1).flatten(0, 1)
    grid_im = grid_image(pairs, nrows=16)
    grid_im.save(save_path)


def grid_image(batch: mx.array, nrows: int):
    B, H, W, _ = batch.shape

    ncols = B // nrows

    grid_im = (batch.reshape(nrows, ncols, H, W, -1) * 255.0).astype(mx.uint8)
    grid_im = grid_im.swapaxes(1, 2).reshape(nrows * H, ncols * W)

    return Image.fromarray(np.array(grid_im))


def generate(model: CVAE, n: int, save_path: Path):
    images = model.sample(n)

    grid_im = grid_image(images, nrows=int(n**0.5))
    grid_im.save(save_path)


def loss_fn(model: CVAE, x: mx.array) -> mx.array:
    x_recon, mu, logvar = model(x)

    # Reconstruction loss
    recon_loss = nn.losses.mse_loss(x_recon, x, reduction="sum")

    # KL divergence
    kl_div = -0.5 * mx.sum(1 + logvar - mu.square() - logvar.exp())

    return recon_loss + kl_div


def train(args):
    image_shape = (64, 64, 1)
    train_dataset, test_dataset = mnist(batch_size=args.batch_size, image_shape=image_shape[:2])

    save_dir = Path(args.save_dir) / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    save_dir.mkdir(parents=True, exist_ok=True)

    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cvae = CVAE(image_shape=image_shape, latent_dim=args.latent_dim, max_filters=args.max_filters)
    mx.eval(cvae.parameters())

    num_params = sum(x.size for _, x in tree_flatten(cvae.trainable_parameters()))
    print(f"Number of trainable params: {num_params / 1e6:0.04f} M")

    optimizer = AdamW(learning_rate=args.lr, weight_decay=args.weight_decay)

    train_vis_batch = next(train_dataset)
    test_vis_batch = next(test_dataset)

    state = [cvae.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(x: mx.array) -> mx.array:
        loss_and_grad_fn = nn.value_and_grad(cvae, loss_fn)
        loss, grad = loss_and_grad_fn(cvae, x)
        optimizer.update(cvae, grad)
        return loss

    pbar = tqdm(total=args.epochs)
    for e in range(1, args.epochs + 1):
        train_dataset.reset()
        cvae.train()

        loss_epoch = 0.0

        for i, batch in enumerate(train_dataset):  # noqa: B007
            x = mx.array(batch["image"])
            loss = step(x)

            # Need to eval state to update model params
            mx.eval(state)

            loss_epoch += loss.item()

        pbar.set_description(f"Epoch {e:4d}, Loss: {loss_epoch / (i + 1):10.2f}")

        cvae.eval()
        reconstruct(cvae, train_vis_batch, images_dir / f"train_recon_{e:03d}.png")
        reconstruct(cvae, test_vis_batch, images_dir / f"test_recon_{e:03d}.png")

        generate(cvae, 64, images_dir / f"generated_samples_{e:03d}.png")

        if e % args.save_interval == 0:
            cvae.save_weights(str(save_dir / f"cvae_{e:03d}.safetensors"))

        pbar.update()

    pbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--cpu", action="store_true", help="Use CPU for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=8, help="Dimension of the latent space")
    parser.add_argument("--max_filters", type=int, default=64, help="Maximum number of filters in the convolutional layers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--save_dir", type=str, default="checkpoints/", help="Directory to save checkpoints")
    parser.add_argument("--save_interval", type=int, default=10, help="Interval to save checkpoints")

    args = parser.parse_args()

    if args.cpu:
        mx.set_default_device(mx.cpu)

    train(args)
