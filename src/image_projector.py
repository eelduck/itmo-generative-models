import torch
from PIL import Image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor, ToPILImage
from tqdm import tqdm

from src.losses import ArcfaceLoss, LpipsLoss, RecLoss, RegLoss
from src.utils import image2e4etensor


class ImageProjector:

    def __init__(
        self, 
        e4e_model, 
        G_model, 
        arcface_model_path=None, 
        use_arcface=False, 
        device=None
    ):
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        self.e4e_model = e4e_model
        self.G = G_model
        self.use_arcface = use_arcface
        self.lpips_loss = LpipsLoss(self.device)
        self.rec_loss = RecLoss()
        self.reg_loss = RegLoss(
            {name: buf for (name, buf) in G_model.synthesis.named_buffers() if "noise_const" in name}
        )
        if self.use_arcface and arcface_model_path:
            self.arcface_loss = ArcfaceLoss(arcface_model_path, self.device)

    def load_and_preprocess_image(self, image_path):
        """Load and preprocess the image."""
        try:
            with Image.open(image_path).convert("RGB") as img:
                preprocess = Compose(
                    [
                        Resize((1024, 1024)),
                        ToTensor(),
                        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                    ]
                )
                return preprocess(img).to(self.device).unsqueeze(0), img
        except Exception as e:
            raise OSError(f"Unable to load image {image_path}: {e}")

    def project_image_to_latent_space(
        self, 
        target_img_dir,
        source_img_dir=None, 
        num_steps=100, 
        learning_rate=0.01,
        reg_w = 5e5,
        rec_w = 0.5,
        lpips_w = 1,
        arcface_w = 1
    ):
        """Project an image to the latent space with optional identity swapping."""
        target_tensor, target_pil = self.load_and_preprocess_image(target_img_dir)
        if source_img_dir:
            source_tensor, _ = self.load_and_preprocess_image(source_img_dir)
        else:
            source_tensor = None

        e4e_tensor = image2e4etensor(target_pil).to(self.device).unsqueeze(0)
        initial_latent = self.e4e_model(e4e_tensor)

        latent_param = torch.nn.Parameter(initial_latent, requires_grad=True)
        optimizer = torch.optim.Adam([latent_param], lr=learning_rate)

        for _ in tqdm(range(num_steps)):
            self.step_optimization_with_optional_swap(
                latent_param, 
                target_tensor, 
                source_tensor, 
                optimizer,
                reg_w,
                rec_w,
                lpips_w,
                arcface_w
            )

        return target_tensor, self.generate_final_tensor(latent_param), latent_param

    def step_optimization_with_optional_swap(
        self, 
        latent_param, 
        target_tensor, 
        source_tensor, 
        optimizer,
        reg_w,
        rec_w,
        lpips_w,
        arcface_w
    ):
        """Perform one step of optimization with optional identity swapping."""
        optimizer.zero_grad()
        synth_tensor = self.G.synthesis(self.broadcast_w_sg(latent_param), noise_mode="const")
        lpips_value = self.lpips_loss(synth_tensor, target_tensor)
        rec_value = self.rec_loss(synth_tensor, target_tensor)
        reg_value = self.reg_loss()

        loss = (lpips_value * lpips_w) + (rec_value * rec_w) + (reg_value * reg_w)

        if self.use_arcface:
            arcface_value = self.arcface_loss(source_tensor, synth_tensor)
            loss += arcface_value * arcface_w

        loss.backward()
        optimizer.step()
        return loss

    def generate_final_tensor(self, latent_param):
        """Generate the final tensor using the adjusted latent parameter."""
        return self.G.synthesis(self.broadcast_w_sg(latent_param), noise_mode="const", force_fp32=True)

    def broadcast_w_sg(self, w_batch, cast_n=18):
        input_ws = []
        for w in w_batch:
            w_broadcast = torch.broadcast_to(w, (cast_n, 512))
            input_ws.append(w_broadcast)
        return torch.stack(input_ws)
