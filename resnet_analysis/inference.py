import os
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
from model import ZPinchResNet

# --- Config ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "zpinch_resnet50.pth"
image_folder = "images"
output_csv = "results.csv"

# --- Transform (same as training) ---
transform = transforms.Compose([
    ResizeKeepAspectPad(target_size=224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# --- Load model ---
model = ZPinchResNet(num_outputs=3)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# --- Inference ---
results = []
for fname in os.listdir(image_folder):
    if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
        continue
    image_path = os.path.join(image_folder, fname)
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
    radius_norm, instability_norm, angle = output[0].cpu().numpy()

    results.append({
        "image": fname,
        "radius": radius_norm * pinch_height,
        "instability": instability_norm * pinch_height,
        "angle": angle
    })

# --- Save CSV ---
print(results)
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved results to {output_csv}")
