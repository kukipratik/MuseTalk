from diffusers import AutoencoderKL
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F  # noqa: F401  # kept for compatibility with any external calls
import cv2
import numpy as np
from PIL import Image  # noqa: F401
import os


class VAE:
    """
    VAE (Variational Autoencoder) class for image processing.
    Loads the Stable Diffusion VAE strictly from a local directory and
    prefers diffusion_pytorch_model.bin over safetensors.
    """

    def __init__(
        self,
        model_path: str = "./models/sd-vae/",
        resized_img: int = 256,
        use_float16: bool = False,
    ):
        """
        :param model_path: Path to the local SD VAE folder (must contain config.json and diffusion_pytorch_model.bin).
        :param resized_img: Target size for image resizing.
        :param use_float16: Whether to use float16 precision for the VAE.
        """
        self.model_path = model_path
        self._resized_img = resized_img

        # Force diffusers to use local files and NOT safetensors.
        # With a local folder containing config.json + diffusion_pytorch_model.bin,
        # this will load the VAE without attempting any network calls or .safetensors.
        self.vae = AutoencoderKL.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_safetensors=False,
            torch_dtype=torch.float16 if use_float16 else torch.float32,
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vae.to(self.device)

        # Track precision mode
        self._use_float16 = bool(use_float16)

        self.scaling_factor = getattr(self.vae.config, "scaling_factor", 0.18215)  # default SD scaling
        self.transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        # Precompute mask once
        self._mask_tensor = self.get_mask_tensor()

    def get_mask_tensor(self) -> torch.Tensor:
        """
        Creates a half-mask tensor for image processing.
        Top half = 1, bottom half = 0.
        """
        mask_tensor = torch.zeros((self._resized_img, self._resized_img))
        mask_tensor[: self._resized_img // 2, :] = 1
        mask_tensor = (mask_tensor >= 0.5).to(mask_tensor.dtype)
        return mask_tensor

    def preprocess_img(self, img_or_path, half_mask: bool = False) -> torch.Tensor:
        """
        Preprocess an image for the VAE.

        :param img_or_path: BGR image (numpy array) or path to image file.
        :param half_mask: Whether to apply a half mask to the image.
        :return: A preprocessed image tensor on the correct device.
        """
        window = []
        if isinstance(img_or_path, str):
            # path
            img = cv2.imread(img_or_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(
                img, (self._resized_img, self._resized_img), interpolation=cv2.INTER_LANCZOS4
            )
            window.append(img)
        else:
            # OLD (assumed BGR)
            # img = cv2.cvtColor(img_or_path, cv2.COLOR_BGR2RGB)

            # NEW (keep as RGB; upstream already RGB)
            img = img_or_path.copy()
            img = cv2.resize(
                img, (self._resized_img, self._resized_img), interpolation=cv2.INTER_LANCZOS4
            )
            window.append(img)

        x = np.asarray(window) / 255.0  # [N, H, W, C]
        x = np.transpose(x, (3, 0, 1, 2))  # [C, N, H, W]
        x = torch.squeeze(torch.from_numpy(x).float())  # [C, H, W]

        if half_mask:
            x = x * (self._mask_tensor > 0.5)

        x = self.transform(x)  # normalize to [-1, 1]
        x = x.unsqueeze(0)  # [1, 3, H, W]
        x = x.to(self.vae.device)
        return x

    @torch.no_grad()
    def encode_latents(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode an image into latent variables.

        :param image: [1, 3, 256, 256] tensor on device
        :return: Latents [1, 4, 32, 32]
        """
        init_latent_dist = self.vae.encode(image.to(self.vae.dtype)).latent_dist
        init_latents = self.scaling_factor * init_latent_dist.sample()
        return init_latents

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> np.ndarray:
        """
        Decode latent variables back into an image (BGR uint8).

        :param latents: [N, 4, 32, 32] latents
        :return: Numpy array [N, H, W, 3] in BGR uint8
        """
        latents = (1 / self.scaling_factor) * latents
        image = self.vae.decode(latents.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).float().numpy()
        image = (image * 255).round().astype("uint8")
        # OLD
        # image = image[..., ::-1]  # RGB -> BGR

        # NEW
        # keep as RGB
        return image

    def get_latents_for_unet(self, img) -> torch.Tensor:
        """
        Prepare latent variables for a U-Net model by concatenating
        masked and unmasked latents along the channel dimension.

        :param img: BGR numpy image or path
        :return: Tensor [1, 8, 32, 32]
        """
        ref_image_masked = self.preprocess_img(img, half_mask=True)
        masked_latents = self.encode_latents(ref_image_masked)

        ref_image_full = self.preprocess_img(img, half_mask=False)
        ref_latents = self.encode_latents(ref_image_full)

        latent_model_input = torch.cat([masked_latents, ref_latents], dim=1)
        return latent_model_input


if __name__ == "__main__":
    # Simple local test
    vae_model_path = "./models/sd-vae/"
    vae = VAE(model_path=vae_model_path, use_float16=False)
    crop_imgs_path = "./results/sun001_crop/"
    latents_out_path = "./results/latents/"
    os.makedirs(latents_out_path, exist_ok=True)

    files = sorted([f for f in os.listdir(crop_imgs_path) if f.lower().endswith(".png")])
    for file in files:
        img_path = os.path.join(crop_imgs_path, file)
        latents = vae.get_latents_for_unet(img_path)
        print(img_path, "latents", latents.size())
        # torch.save(latents, os.path.join(latents_out_path, file.replace(".png", ".pt")))
