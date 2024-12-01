import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite multiclass hair segmentation model
interpreter = tf.lite.Interpreter(model_path="selfie_multiclass_256x256.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model's expected input details
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
print("Model's expected input shape:", input_shape)
print("Model's expected input dtype:", input_dtype)

# Define the hair class index (you need to set this based on your model)
hair_class_index = 1  # Replace with the actual index for hair in your model

# Open the default webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    input_height, input_width = input_shape[1], input_shape[2]
    resized_frame = cv2.resize(frame, (input_width, input_height))

    # Convert the image to RGB if needed (most models expect RGB)
    input_data = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Normalize the image if required (e.g., scale pixel values to [0, 1])
    input_data = input_data.astype(np.float32) / 255.0

    # Add a batch dimension
    input_data = np.expand_dims(input_data, axis=0)

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Debug: Print output data shape and dtype
    # print("Output data shape:", output_data.shape)
    # print("Output data dtype:", output_data.dtype)

    # Remove batch dimension if present
    output_data = np.squeeze(output_data)

    # For multiclass segmentation, use argmax to get the class labels
    class_map = np.argmax(output_data, axis=-1).astype(np.uint8)  # Shape: (height, width)

    # Create a mask for the hair class
    hair_mask = (class_map == hair_class_index).astype(np.uint8) * 255  # Convert to 0 or 255

    # Resize the mask back to the original frame size
    hair_mask = cv2.resize(hair_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Optional: Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    hair_mask = cv2.morphologyEx(hair_mask, cv2.MORPH_CLOSE, kernel)

    # Create a colored overlay for the hair region
    colored_hair = np.zeros_like(frame)
    colored_hair[:, :] = [255, 0, 0]  # Blue color in BGR

    # Apply the mask to create the hair overlay
    hair_region = cv2.bitwise_and(colored_hair, colored_hair, mask=hair_mask)

    # Combine the hair overlay with the original frame
    background = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(hair_mask))
    output_frame = cv2.addWeighted(background, 1, hair_region, 0.7, 0)

    # Display the result
    cv2.imshow('Real-time Hair Segmentation', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()