import cv2
import torch
import numpy as np
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
import mediapipe as mp
import time

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

# Load multiple hair texture images
texture_paths = ['hair_texture1.jpg', 'hair_texture2.jpg', 'hair_texture3.jpg']  # Replace with your texture image paths
textures = [cv2.imread(path) for path in texture_paths]
if any(tex is None for tex in textures):
    print("Error: Could not load one or more texture images.")
    exit()
texture_index = 0  # Start with the first texture

# Create thumbnails of textures
thumb_size = (100, 100)
thumbnails = [cv2.resize(tex, thumb_size) for tex in textures]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Define arrow positions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
arrow_size = 50
left_arrow_pos = (50, frame_height // 2)
right_arrow_pos = (frame_width - 100, frame_height // 2)

# Variables for hold time feature
hold_start_time = None
hold_duration_required = 1.0  # seconds

# Variables to track if hand is over arrow
hand_over_left_arrow = False
hand_over_right_arrow = False
hold_position = None

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break

        # Flip the frame horizontally for natural interaction
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Preprocess the image for segmentation
        inputs = processor(images=image_rgb, return_tensors="pt")
        inputs = inputs.to(device)

        # Forward pass for segmentation
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            upsampled_logits = torch.nn.functional.interpolate(
                logits,
                size=image_rgb.shape[:2],
                mode='bilinear',
                align_corners=False
            )

            # Get the segmentation map
            segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create a mask where hair pixels are 1, others are 0
        hair_mask = (segmentation == hair_class_index).astype(np.uint8) * 255  # Multiply by 255 for proper masking

        # Resize the current texture to the frame size
        texture_resized = cv2.resize(textures[texture_index], (frame.shape[1], frame.shape[0]))

        # Apply the mask to the texture
        hair_texture_applied = cv2.bitwise_and(texture_resized, texture_resized, mask=hair_mask)

        # Invert the hair mask to get the background
        inv_hair_mask = cv2.bitwise_not(hair_mask)

        # Apply the inverted mask to the original frame to get background without hair
        background = cv2.bitwise_and(frame, frame, mask=inv_hair_mask)

        # Combine the background and the textured hair
        combined = cv2.add(background, hair_texture_applied)

        # Hand detection with MediaPipe
        result = hands.process(image_rgb)

        previous_hand_over_left_arrow = hand_over_left_arrow
        previous_hand_over_right_arrow = hand_over_right_arrow
        hand_over_left_arrow = False
        hand_over_right_arrow = False

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # Get the bounding box of the hand
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                x_min = int(min(x_coords) * frame.shape[1])
                x_max = int(max(x_coords) * frame.shape[1])
                y_min = int(min(y_coords) * frame.shape[0])
                y_max = int(max(y_coords) * frame.shape[0])

                # Draw hand bounding box
                cv2.rectangle(combined, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Check if hand is over left arrow
                if (x_min < left_arrow_pos[0] + arrow_size and x_max > left_arrow_pos[0] - arrow_size and
                    y_min < left_arrow_pos[1] + arrow_size and y_max > left_arrow_pos[1] - arrow_size):
                    hand_over_left_arrow = True

                # Check if hand is over right arrow
                if (x_min < right_arrow_pos[0] + arrow_size and x_max > right_arrow_pos[0] - arrow_size and
                    y_min < right_arrow_pos[1] + arrow_size and y_max > right_arrow_pos[1] - arrow_size):
                    hand_over_right_arrow = True

        # Switch textures if hand is over an arrow for required duration
        current_time = time.time()
        if hand_over_left_arrow or hand_over_right_arrow:
            if hold_start_time is None:
                hold_start_time = current_time
                hold_position = 'left' if hand_over_left_arrow else 'right'
            elif hold_position == ('left' if hand_over_left_arrow else 'right'):
                elapsed_time = current_time - hold_start_time
                if elapsed_time >= hold_duration_required:
                    if hold_position == 'left':
                        texture_index = (texture_index - 1) % len(textures)
                    elif hold_position == 'right':
                        texture_index = (texture_index + 1) % len(textures)
                    hold_start_time = None  # Reset hold start time after switching
                    hold_position = None
            else:
                # Hand moved to the other arrow, reset the timer
                hold_start_time = current_time
                hold_position = 'left' if hand_over_left_arrow else 'right'
        else:
            hold_start_time = None  # Reset if hand is not over any arrow
            hold_position = None

        # Draw arrows on the combined image
        # Left Arrow
        cv2.arrowedLine(combined,
                        (left_arrow_pos[0] + arrow_size, left_arrow_pos[1]),
                        (left_arrow_pos[0] - arrow_size, left_arrow_pos[1]),
                        (255, 0, 0), 5)
        # Right Arrow
        cv2.arrowedLine(combined,
                        (right_arrow_pos[0] - arrow_size, right_arrow_pos[1]),
                        (right_arrow_pos[0] + arrow_size, right_arrow_pos[1]),
                        (255, 0, 0), 5)

        # Highlight arrow if hand is over it and show hold progress
        if hand_over_left_arrow:
            cv2.circle(combined, left_arrow_pos, arrow_size, (0, 255, 0), 2)
            # Show hold progress
            if hold_start_time is not None and hold_position == 'left':
                progress = (current_time - hold_start_time) / hold_duration_required
                progress = min(max(progress, 0), 1)  # Clamp between 0 and 1
                progress_radius = int(arrow_size * progress)
                cv2.circle(combined, left_arrow_pos, progress_radius, (0, 255, 0), -1)
        if hand_over_right_arrow:
            cv2.circle(combined, right_arrow_pos, arrow_size, (0, 255, 0), 2)
            # Show hold progress
            if hold_start_time is not None and hold_position == 'right':
                progress = (current_time - hold_start_time) / hold_duration_required
                progress = min(max(progress, 0), 1)  # Clamp between 0 and 1
                progress_radius = int(arrow_size * progress)
                cv2.circle(combined, right_arrow_pos, progress_radius, (0, 255, 0), -1)

        # Display thumbnails of textures
        thumbnail_y = frame_height - thumb_size[1] - 10  # 10 pixels from bottom
        for idx, thumbnail in enumerate(thumbnails):
            thumbnail_x = 10 + idx * (thumb_size[0] + 10)  # 10 pixels between thumbnails
            if idx == texture_index:
                # Highlight the selected texture
                cv2.rectangle(combined, (thumbnail_x - 5, thumbnail_y - 5), (thumbnail_x + thumb_size[0] + 5, thumbnail_y + thumb_size[1] + 5), (0, 255, 0), 2)
            combined[thumbnail_y:thumbnail_y + thumb_size[1], thumbnail_x:thumbnail_x + thumb_size[0]] = thumbnail

        # Display the resulting frame
        cv2.imshow('Real-time Hair Texture Application', combined)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    hands.close()