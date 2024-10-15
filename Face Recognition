import face_recognition
import cv2
import numpy as np

# Load known images and encode them
def load_known_faces(known_face_names, known_face_paths):
    known_faces = []
    for name, path in zip(known_face_names, known_face_paths):
        image = face_recognition.load_image_file(path)
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append((name, encoding))
    return known_faces

# Recognize faces in the provided image
def recognize_faces(image, known_faces):
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_image)
    face_encodings = face_recognition.face_encodings(rgb_image, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces([encoding for name, encoding in known_faces], face_encoding)
        name = "Unknown"

        # Use the known face with the smallest distance
        face_distances = face_recognition.face_distance([encoding for name, encoding in known_faces], face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_faces[best_match_index][0]

        # Draw a rectangle around the face and label it
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    return image

# Main function to run the facial recognition
def main():
    # Load known faces
    known_face_names = ["Alice", "Bob"]
    known_face_paths = ["path/to/alice.jpg", "path/to/bob.jpg"]  # Replace with actual image paths
    known_faces = load_known_faces(known_face_names, known_face_paths)

    # Load an image to recognize faces
    test_image = cv2.imread("path/to/test_image.jpg")  # Replace with an actual test image path
    recognized_image = recognize_faces(test_image, known_faces)

    # Display the result
    cv2.imshow("Facial Recognition", recognized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
