import h5py
from torchcodec.decoders import VideoDecoder
import matplotlib.pyplot as plt

# Ton code existant
decoder = VideoDecoder("/home/cl-ment-prigent/LAM/713.mp4", device='cpu')
img = decoder[0]

print(f"Shape originale : {img.shape}") # Probablement (C, H, W) ou (H, W, C)


img_cpu = img.cpu()

if img_cpu.shape[0] == 3: 
    img_to_show = img_cpu.permute(1, 2, 0).numpy()
else:
    img_to_show = img_cpu.numpy()

plt.imshow(img_to_show)
plt.axis('off') # Cache les axes X et Y
plt.title("Première frame du projet LAM")
plt.show()