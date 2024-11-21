import mlx.core as mx
import mlx.nn as nn


def upsample_nearest(x, scale: int = 2):
    B, H, W, C = x.shape

    # (B, H, W, C) -> (B, H, 1, W, 1, C)
    x = mx.broadcast_to(x[:, :, None, :, None, :], (B, H, scale, W, scale, C))

    # (B, H, 1, W, 1, C) -> (B, H * scale, W * scale, C)
    x = x.reshape(B, H * scale, W * scale, C)
    return x


class UpsamplingConv2d(nn.Module):
    """
    A convolutional layer that upsamples the input by a factor of 2. MLX does
    not yet support transposed convolutions, so we approximate them with
    nearest neighbor upsampling followed by a convolution. This is similar to
    the approach used in the original U-Net.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        scale: int = 2,
    ):
        super().__init__()
        self.scale = scale
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def __call__(self, x):
        x = self.conv(upsample_nearest(x, self.scale))
        return x


class Encoder(nn.Module):
    """Encoder network for CVAE.

    Maps input to latent space.
    """

    def __init__(self, image_shape: tuple[int, int, int], latent_dim: int, max_filters: int):
        """Initializes the encoder network.

        Args:
            - in_shape: input shape (H, W, C)
            - latent_dim: dimension of the latent space
            - max_filters: maximum number of filters in the network
        """
        super().__init__()

        H, W, C = image_shape

        n_filters_1 = max_filters // 4
        n_filters_2 = max_filters // 2
        n_filters_3 = max_filters

        # (B, H, W, C) -> (B, H/2, W/2, n_filters_1)
        self.conv1 = nn.Conv2d(C, n_filters_1, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters_1)

        # (B, H/2, W/2, n_filters_1) -> (B, H/4, W/4, n_filters_2)
        self.conv2 = nn.Conv2d(n_filters_1, n_filters_2, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters_2)

        # (B, H/4, W/4, n_filters_2) -> (B, H/8, W/8, n_filters_3)
        self.conv3 = nn.Conv2d(n_filters_2, n_filters_3, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(n_filters_3)

        out_shape = (H // 8, W // 8, n_filters_3)
        flattened_dim = out_shape[0] * out_shape[1] * out_shape[2]

        self.proj_mean = nn.Linear(flattened_dim, latent_dim)
        self.proj_logvar = nn.Linear(flattened_dim, latent_dim)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        # (B, H, W, C) -> (B, H/2, W/2, n_filters_1)
        x = nn.leaky_relu(self.bn1(self.conv1(x)))
        # (B, H/2, W/2, n_filters_1) -> (B, H/4, W/4, n_filters_2)
        x = nn.leaky_relu(self.bn2(self.conv2(x)))
        # (B, H/4, W/4, n_filters_2) -> (B, H/8, W/8, n_filters_3)
        x = nn.leaky_relu(self.bn3(self.conv3(x)))
        # (B, H/8, W/8, n_filters_3) -> (B, n_filters_3 * H/8 * W/8)
        x = mx.flatten(x, start_axis=1)

        # (B, n_filters_3 * H/8 * W/8) -> (B, latent_dim)
        mean = self.proj_mean(x)
        logvar = self.proj_logvar(x)

        # Compute standard deviation from log variance
        sigma = mx.exp(0.5 * logvar)

        # Sample random values from normal distribution
        eps = mx.random.normal(sigma.shape)

        # Reparameterization trick
        z = mean + sigma * eps

        return z, mean, logvar


class Decoder(nn.Module):
    """Decoder network for CVAE.

    Maps latent space to output space.
    """

    def __init__(self, image_shape: tuple[int, int, int], latent_dim: int, max_filters: int):
        super().__init__()

        H, W, C = image_shape

        n_filters_3 = max_filters
        n_filters_2 = max_filters // 2
        n_filters_1 = max_filters // 4

        self.in_shape = (H // 8, W // 8, n_filters_3)
        flattened_dim = self.in_shape[0] * self.in_shape[1] * self.in_shape[2]

        # (B, latent_dim) -> (B, n_filters_3 * H/8 * W/8)
        self.proj = nn.Linear(latent_dim, flattened_dim)

        # (B, H/8, W/8, n_filters_3) -> (B, H/4, W/4, n_filters_2)
        self.upconv1 = UpsamplingConv2d(n_filters_3, n_filters_2, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(n_filters_2)

        # (B, H/4, W/4, n_filters_2) -> (B, H/2, W/2, n_filters_1)
        self.upconv2 = UpsamplingConv2d(n_filters_2, n_filters_1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(n_filters_1)

        # (B, H/2, W/2, n_filters_1) -> (B, H, W, C)
        self.upconv3 = UpsamplingConv2d(n_filters_1, C, kernel_size=3, stride=1, padding=1)

    def __call__(self, z: mx.array):
        # (B, latent_dim) -> (B, n_filters_3 * H/8 * W/8)
        x = self.proj(z)

        # (B, n_filters_3 * H/8 * W/8) -> (B, H/8, W/8, n_filters_3)
        x = x.reshape(-1, *self.in_shape)

        # (B, H/8, W/8, n_filters_3) -> (B, H/4, W/4, n_filters_2)
        x = nn.leaky_relu(self.bn1(self.upconv1(x)))

        # (B, H/4, W/4, n_filters_2) -> (B, H/2, W/2, n_filters_1)
        x = nn.leaky_relu(self.bn2(self.upconv2(x)))

        # Apply sigmoid activation to get pixel values in [0, 1]
        x = mx.sigmoid(self.upconv3(x))

        return x


class CVAE(nn.Module):
    """Convolutional Variational Autoencoder."""

    def __init__(self, image_shape: tuple[int, int, int], latent_dim: int, max_filters: int):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(image_shape, latent_dim, max_filters)
        self.decoder = Decoder(image_shape, latent_dim, max_filters)

    def __call__(self, x: mx.array):
        z, mean, logvar = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    def encode(self, x: mx.array):
        return self.encoder(x)[0]

    def decode(self, z: mx.array):
        return self.decoder(z)

    def sample(self, num_samples: int):
        # Sample random values from standard normal distribution
        z = mx.random.normal((num_samples, self.latent_dim))
        return self.decode(z)
