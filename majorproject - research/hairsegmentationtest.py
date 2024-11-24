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
            logits = outputs.logits  # shape: [batch_size, num_classes, height, width]

            # Resize logits to original image size
            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image.shape[:2],  # Corrected here
                mode='bilinear',
                align_corners=False
            )

            # Get the segmentation map
            segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create a mask where hair pixels are 1, others are 0
        hair_mask = (segmentation == hair_class_index).astype(np.uint8)

        # Create a color mask (e.g., red color for hair regions)
        hair_mask_colored = np.zeros_like(frame)
        hair_mask_colored[:, :, 2] = hair_mask * 255  # Red channel

        # Overlay the mask on the original frame
        overlay = cv2.addWeighted(frame, 0.7, hair_mask_colored, 0.3, 0)

        # Display the resulting frame
        cv2.imshow('Real-time Hair Segmentation', overlay)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()