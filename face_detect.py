import cv2
import time
import os

# Folder to save detected faces
save_folder = "detected_faces"
os.makedirs(save_folder, exist_ok=True)

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt.xml")

if face_cascade.empty():
    print("Error loading cascade classifier")
    exit()

# Open the default camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame from webcam
    r, frame = cap.read()
    
    if not r or frame is None:
        continue  # Skip processing if frame is empty
    
    # Resize frame for faster processing (Adjust scale as needed)
    scale_percent = 50  # Reduce frame size to 50% of original
    frame = cv2.resize(frame, (0, 0), fx=scale_percent/100, fy=scale_percent/100)

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces with optimized parameters
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=4)

    # Draw rectangles and save cropped faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 125, 0), 2)
        cv2.putText(frame, "DETECTED", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 125, 255), 2, cv2.LINE_AA)

        # Save cropped face image
        face_crop = frame[y:y+h, x:x+w]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(save_folder, f"face_{timestamp}.jpg")
        cv2.imwrite(filename, face_crop)
        print(f"Saved: {filename}")

    # Display the frame
    cv2.imshow("Face Detection", frame)

    # Break loop on 'n' key press
    if cv2.waitKey(1) & 0xFF == ord('n'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
