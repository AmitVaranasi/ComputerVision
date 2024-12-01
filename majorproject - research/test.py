import cv2
import numpy as np
import tensorflow as tf

# Load the TFLite hair segmentation model
interpreter = tf.lite.Interpreter(model_path="hair_segmenter.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Print model's expected input details
input_shape = input_details[0]['shape']
print("Model's expected input shape:", input_shape)
print("Model's expected input dtype:", input_details[0]['dtype'])

# Open the default webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_height, input_width = input_shape[1], input_shape[2]

    resized_frame = cv2.resize(frame, (input_width, input_height))

    # Convert BGR to RGBA
    input_data = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGBA)

    # Expand dimensions to match the model's input shape
    input_data = np.expand_dims(input_data, axis=0)

    # Normalize pixel values if required
    input_data = input_data.astype(np.float32) / 255.0

    # Verify the input data shape
    print("Input data shape:", input_data.shape)

    # Set the tensor to the input data
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the results
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Debugging: Check output data shape and type
    print("Output data shape:", output_data.shape, "dtype:", output_data.dtype)

    # Process the output_data to create a mask
    # Modify based on your model's output format
    if output_data.ndim == 3 and output_data.shape[-1] > 1:
        # Multi-class segmentation
        class_map = np.argmax(output_data, axis=-1).astype(np.uint8)
        # Replace 'hair_class_index' with the correct index for hair
        hair_class_index = 1  # Example index
        hair_mask = (class_map == hair_class_index).astype(np.uint8) * 255
    else:
        # Binary segmentation
        hair_mask = output_data.squeeze() > 0.5  # Adjust threshold if needed
        hair_mask = hair_mask.astype(np.uint8) * 255

    # Resize the mask to match the frame size
    hair_mask = cv2.resize(hair_mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Apply the mask to the frame
    hair_mask_3d = cv2.merge([hair_mask, hair_mask, hair_mask])

    # Verify shapes and data types
    print("Frame shape:", frame.shape, "dtype:", frame.dtype)
    print("Hair mask 3D shape:", hair_mask_3d.shape, "dtype:", hair_mask_3d.dtype)

    # Ensure data types match
    frame = frame.astype(np.uint8)
    hair_mask_3d = hair_mask_3d.astype(np.uint8)

    # Perform bitwise AND operation
    hair_region = cv2.bitwise_and(frame, hair_mask_3d)

    # Optionally, change hair color
    hair_region[hair_mask == 255] = [255, 0, 0]  # Blue color in BGR

    # Combine hair region with the original frame
    output_frame = np.where(hair_mask_3d == 255, hair_region, frame)

    # Display the result
    cv2.imshow('Real-time Hair Segmentation', output_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()