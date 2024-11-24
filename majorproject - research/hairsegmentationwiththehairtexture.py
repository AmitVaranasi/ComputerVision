import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Load the model and processor
processor = AutoImageProcessor.from_pretrained("isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing")
model = SegformerForSemanticSegmentation.from_pretrained("isjackwild/segformer-b0-finetuned-segments-skin-hair-clothing")

# Set the model to evaluation mode
model.eval()

# Use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Get the class index for 'hair'
hair_class_index = None
for idx, label in model.config.id2label.items():
    if label.lower() == 'hair':
        hair_class_index = int(idx)
        break

if hair_class_index is None:
    print("Error: 'hair' class not found in model labels.")
    exit()

# Load the hair texture image
texture = cv2.imread('hair_texture.jpg')  # Replace with your texture image path
if texture is None:
    print("Error: Could not load texture image.")
    exit()

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Resize the frame for faster processing (optional)
        # frame = cv2.resize(frame, (640, 480))

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the image
        inputs = processor(images=image, return_tensors="pt")
        inputs = inputs.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            # Resize logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.shape[:2],
                mode='bilinear',
                align_corners=False
            )

            # Get the segmentation map
            segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create a mask where hair pixels are 1, others are 0
        hair_mask = (segmentation == hair_class_index).astype(np.uint8) * 255  # Multiply by 255 for proper masking

        # Resize the texture to the frame size
        texture_resized = cv2.resize(texture, (frame.shape[1], frame.shape[0]))

        # Apply the mask to the texture
        hair_texture_applied = cv2.bitwise_and(texture_resized, texture_resized, mask=hair_mask)

        # Invert the hair mask to get the background
        inv_hair_mask = cv2.bitwise_not(hair_mask)

        # Apply the inverted mask to the original frame to get background without hair
        background = cv2.bitwise_and(frame, frame, mask=inv_hair_mask)

        # Combine the background and the textured hair
        combined = cv2.add(background, hair_texture_applied)

        # Display the resulting frame
        cv2.imshow('Real-time Hair Texture Application', combined)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()