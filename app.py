import streamlit as st
import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
import time

# ====== Generator definition ======
class Generator(nn.Module):
    def __init__(self, latent_dim=100, n_classes=10, img_shape=(1, 28, 28)):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Linear(1024, 784),
            nn.Tanh()
        )
        self.img_shape = img_shape

    def forward(self, noise, labels):
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *self.img_shape)
        return img
    
# ====== Load trained generator ======
@st.cache_resource
def load_generator():
    model = Generator()
    model.load_state_dict(torch.load("mnist_cgan_generator.pth", map_location="cpu"))
    model.eval()
    return model

# ====== Generate Images ======
def generate_images(generator, digit, n_samples=5):
    seed = int((time.time() * 1000) % 100000)  # unique per generation
    torch.manual_seed(seed)
    z = torch.randn(n_samples, 100)
    labels = torch.full((n_samples,), digit, dtype=torch.long)
    with torch.no_grad():
        images = generator(z, labels).cpu()
    return images

# ====== Streamlit UI ======
st.title("🖊️ Handwritten Digit Image Generator")
st.markdown("Generate synthetic MNIST-like images.")

digit = st.selectbox("Choose a digit to generate (0-9):", list(range(10)), index=0)

if st.button("🎨 Generate Images"):
    generator = load_generator()
    images = generate_images(generator, digit)

    st.markdown(f"### Generated images of digit {digit}")
    cols = st.columns(5)
    for i in range(5):
        img_np = images[i][0].numpy()
        fig, ax = plt.subplots()
        ax.imshow(img_np, cmap='gray')
        ax.axis("off")
        cols[i].pyplot(fig)
        cols[i].caption(f"Sample {i+1}")
