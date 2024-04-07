import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from PIL import Image


def image2tensor_norm(image):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    tensor = transform(image)
    return tensor


# image = ((image * std) + mean)
def plot_image(tensor, plot_size=(8, 8)):
    img = get_image(tensor)
    plt.rcParams["figure.figsize"] = plot_size
    plt.imshow(img)

def get_image(tensor):
    tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(tensor[0].cpu().numpy(), "RGB")

def broadcast_w_sg(w_batch, cast_n=18):
    input_ws = []
    for w in w_batch:
        w_broadcast = torch.broadcast_to(w, (cast_n, 512))
        input_ws.append(w_broadcast)
    return torch.stack(input_ws)

def interpolate(latent1, latent2, G, psi=0.5, indeces=[i for i in range(0,18)]):
    init_latent = latent1.clone()
    latent1 = latent1.clone()
    latent2 = latent2.clone()
    for i in indeces:
        latent1[:, i] = latent2[:, i].lerp(latent1[:, i], psi)

    edited_tensor = G.synthesis(latent1, noise_mode='const', force_fp32=True)
    init_tensor = G.synthesis(init_latent, noise_mode='const', force_fp32=True)
    return edited_tensor