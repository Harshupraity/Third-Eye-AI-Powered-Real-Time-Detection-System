import face_recognition
import pickle
import os

# Folder where the known faces are stored
folder_path = os.path.expanduser("~/third_eye_project/known_faces")

# Check if the folder exists
if not os.path.exists(folder_path):
    raise ValueError(f"The folder {folder_path} does not exist.")

# Initialize lists to store face encodings and corresponding names
known_face_encodings = []
known_face_names = []

# Loop through all files in the known_faces directory
for filename in os.listdir(folder_path):
    if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(folder_path, filename)
        image = face_recognition.load_image_file(image_path)
        
        # Get face encodings
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            # Assuming the first encoding is the correct one (there should be only one face in each image)
            encoding = encodings[0]
            name = filename.split(".")[0]  # Use the filename (e.g., person1) as the name
            
            # Append the encoding and name
            known_face_encodings.append(encoding)
            known_face_names.append(name)

# Save the known face encodings to a pickle file
pickle_file = os.path.expanduser("~/third_eye_project/face_encodings.pkl")
with open(pickle_file, "wb") as f:
    pickle.dump(dict(zip(known_face_names, known_face_encodings)), f)

print(f"face_encodings.pkl file created successfully at {pickle_file}")
