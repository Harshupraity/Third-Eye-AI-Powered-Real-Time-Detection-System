import face_recognition
import pickle
import os

# Directory where your images of known people are stored
image_directory = '/home/devesh/third_eye_project/known_faces/'

# List of image files for known people (these should be images of people you want to recognize)
image_files = os.listdir(image_directory)

# Dictionary to store known face encodings
known_faces = {}

# Loop through each image file and extract face encodings
for image_file in image_files:
    image_path = os.path.join(image_directory, image_file)
    try:
        # Load the image
        image = face_recognition.load_image_file(image_path)
        
        # Extract face encodings (if any faces are found in the image)
        encodings = face_recognition.face_encodings(image)
        
        # If encodings are found, save the first one to the dictionary with the name (image file name)
        if encodings:
            name = os.path.splitext(image_file)[0]  # Use the file name (without extension) as the name
            known_faces[name] = encodings[0]
            print(f"Encoding for {name} saved.")
    except Exception as e:
        print(f"Error processing {image_file}: {e}")

# Save the known faces dictionary to a pickle file
with open('face_encodings.pkl', 'wb') as f:
    pickle.dump(known_faces, f)

print("Face encodings saved to face_encodings.pkl")
