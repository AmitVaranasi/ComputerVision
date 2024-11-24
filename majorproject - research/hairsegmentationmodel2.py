# Import necessary libraries
import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
from PIL import Image

# Load the pre-trained model and processor
processor = AutoImageProcessor.from_pretrained("Allison/segformer-hair-segmentation-10k-steps")
model = SegformerForSemanticSegmentation.from_pretrained("Allison/segformer-hair-segmentation-10k-steps")

# Set the model to evaluation mode and move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Initialize webcam capture
cap = cv2.VideoCapture(1)  # Use 0 for the default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Preprocess the image
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get the segmentation map
    logits = outputs.logits
    upsampled_logits = torch.nn.functional.interpolate(
        logits,
        size=pil_image.size[::-1],  # (height, width)
        mode='bilinear',
        align_corners=False
    )
    segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

    # Create a mask where hair pixels are set to 255
    mask = (segmentation == 1).astype(np.uint8) * 255

    # Apply the mask to the original frame
    masked_frame = cv2.bitwise_and(frame, frame, mask=mask)

    # Display the original frame and the masked frame side by side
    combined_frame = np.hstack((frame, masked_frame))
    cv2.imshow('Original and Hair Segmentation', combined_frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()