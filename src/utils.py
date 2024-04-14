import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
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

def plot_images_in_row(images):
    """ Plot a list of tensors in a row """
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, image in zip(axes, images):
        ax.imshow(image)
        ax.axis('off')
    plt.show()

def get_image(tensor):
    tensor = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    return Image.fromarray(tensor[0].cpu().numpy(), "RGB")


def broadcast_w_sg(w_batch, cast_n=18):
    input_ws = []
    for w in w_batch:
        w_broadcast = torch.broadcast_to(w, (cast_n, 512))
        input_ws.append(w_broadcast)
    return torch.stack(input_ws)


def interpolate(latent1, latent2, G, psi=0.5, indeces=[i for i in range(0, 18)]):
    latent1 = latent1.clone()
    latent2 = latent2.clone()
    for i in indeces:
        latent1[:, i] = latent2[:, i].lerp(latent1[:, i], psi)

    edited_tensor = G.synthesis(latent1, noise_mode="const", force_fp32=True)
    return edited_tensor


image2e4etensor = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Resize((256, 256)), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
)
