import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Load the hair segmentation model
processor = AutoImageProcessor.from_pretrained("isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing")
model = SegformerForSemanticSegmentation.from_pretrained("isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing")
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the source image
source_image = cv2.imread('Man-Bun-Undercut.jpg')  # Replace with your source image path
if source_image is None:
    print("Error: Could not load source image.")
    exit()

# Convert BGR to RGB
image_rgb = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)

# Preprocess the image
inputs = processor(images=image_rgb, return_tensors="pt")
inputs = inputs.to(device)

# Forward pass for segmentation
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

    # Resize logits to original image size
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=image_rgb.shape[:2],
        mode='bilinear',
        align_corners=False
    )

    # Get the segmentation map
    segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

# Get the class index for 'hair'
hair_class_index = None
for idx, label in model.config.id2label.items():
    if label.lower() == 'hair':
        hair_class_index = int(idx)
        break

if hair_class_index is None:
    print("Error: 'hair' class not found in model labels.")
    exit()

# Create a mask where hair pixels are 255, others are 0
hair_mask = (segmentation == hair_class_index).astype(np.uint8) * 255

# Optional: Apply some morphological operations to clean up the mask
kernel = np.ones((5, 5), np.uint8)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)
hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_OPEN, kernel)

# Save or display the hair mask
# Ensure the source image has an alpha channel
source_image_rgba = cv2.cvtColor(source_image, cv2.COLOR_BGR2BGRA)

# Set the alpha channel based on the hair mask
source_image_rgba[:, :, 3] = hair_mask

# Save the hairstyle image with transparency
cv2.imwrite('extracted_hairstyle3.png', source_image_rgba)