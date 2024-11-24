import cv2
import numpy as np
import mediapipe as mp

def overlay_transparent(background, overlay, x, y):
    """
    Overlays a transparent PNG onto another image using CV2.
    :param background: The background image (OpenCV image).
    :param overlay: The transparent image to overlay (with alpha channel).
    :param x: The x-coordinate where the overlay is placed.
    :param y: The y-coordinate where the overlay is placed.
    """
    # Get dimensions
    h_bg, w_bg = background.shape[:2]
    h_ol, w_ol = overlay.shape[:2]

    # Ensure the overlay is within the frame boundaries
    if x >= w_bg or y >= h_bg:
        return

    # Clip the overlay if it goes beyond the background dimensions
    overlay = overlay[max(0, -y):min(h_ol, h_bg - y), max(0, -x):min(w_ol, w_bg - x)]
    h_ol, w_ol = overlay.shape[:2]

    if h_ol <= 0 or w_ol <= 0:
        return

    # Adjust x and y if they are negative
    x = max(x, 0)
    y = max(y, 0)

    # Split the overlay image into BGR and Alpha channels
    if overlay.shape[2] == 4:
        overlay_img = overlay[:, :, :3]
        mask = overlay[:, :, 3:] / 255.0
    else:
        overlay_img = overlay
        mask = np.ones((h_ol, w_ol, 1), dtype=np.float32)

    # Extract the region of interest from the background
    roi = background[y:y+h_ol, x:x+w_ol]

    # Blend the overlay with the ROI
    roi = (1.0 - mask) * roi + mask * overlay_img

    # Put the blended ROI back into the frame
    background[y:y+h_ol, x:x+w_ol] = roi

def main():
    # Initialize Mediapipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    # Load the hairstyle image with alpha channel
    hairstyle = cv2.imread('extracted_hairstyle.png', cv2.IMREAD_UNCHANGED)
    if hairstyle is None:
        print("Error: Hairstyle image not found.")
        return

    # Start video capture
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame horizontally for a mirror-like effect
        frame = cv2.flip(frame, 1)
        img_h, img_w = frame.shape[:2]

        # Convert the BGR image to RGB before processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and find face landmarks
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Convert landmarks to pixel coordinates
                landmark_points = np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks])

                # Indices for the top of the forehead and chin
                forehead_idx = 10  # Top of forehead
                chin_idx = 152     # Chin

                # Get the coordinates
                forehead_point = landmark_points[forehead_idx]
                chin_point = landmark_points[chin_idx]

                # Calculate face height
                face_height = np.linalg.norm(forehead_point - chin_point)

                # Estimate the width of the face
                left_cheek_idx = 234
                right_cheek_idx = 454
                left_cheek_point = landmark_points[left_cheek_idx]
                right_cheek_point = landmark_points[right_cheek_idx]
                face_width = np.linalg.norm(left_cheek_point - right_cheek_point)

                # Set the size for the hairstyle overlay
                overlay_height = int(face_height * 1.5)  # Adjust multiplier as needed
                overlay_width = int(face_width * 1.2)    # Adjust multiplier as needed
                overlay_size = (overlay_width, overlay_height)

                # Calculate the position to place the overlay
                x = int(forehead_point[0] - overlay_width / 2)
                y = int(forehead_point[1] - overlay_height * 0.8)  # Adjust as needed

                # Resize the hairstyle image
                resized_hairstyle = cv2.resize(hairstyle, overlay_size, interpolation=cv2.INTER_AREA)

                # Overlay the hairstyle
                overlay_transparent(frame, resized_hairstyle, x, y)

        # Display the result
        cv2.imshow('Hairstyle Overlay', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()