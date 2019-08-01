import face_recognition
from PIL import Image, ImageDraw
import cv2
import numpy as np
import os

class Person:
    def __init__(self, name, images):
        self.name = name
        self.images = images
        
def read_dirs(path):
    dirs = []

    for (dirpath, dirnames, filenames) in os.walk(path):
        dirs.append(dirpath)

    return dirs
    
def get_img(path, dir_path):
    print('began process...')
    img = Image.open(path)
    w, h = img.size
    print(w, h)
    #resizing img
    ratio = (320 / float(img.size[0]))
    print(ratio)
    hsize = int((float(h) / float(ratio)))
    wsize = int((float(w) / float(ratio)))
    img = img.resize((wsize, hsize), Image.ANTIALIAS)
    image_title = dir_path + "/resized_image.jpg"
    #os.remove(path)
    img.save(image_title)
    print("person has been loaded & resized")
    img = Image.open(image_title)
    new_image = face_recognition.load_image_file(image_title)
    new_face_encoding = face_recognition.face_encodings(new_image)[0]
    print("person learned")
    return (img,new_face_encoding)

    
def get_people(dirs):
    people = []
    
    for i in dirs:
        img_paths = [(i + '/' + f) for f in os.listdir(i) \
                     if os.path.isfile(os.path.join(i, f))]
        img_enc = []

        for path in img_paths:
            img_enc.append(get_img(path, i))

        people.append(Person(i[10:], img_enc))
        
    return people

dirname = os.path.dirname(__file__)
img_path = os.path.join(dirname, '../images')
dirs = read_dirs(img_path)
people = get_people(dirs)

video_capture = cv2.VideoCapture(0)

known_face_encodings=[]
known_face_names=[]

for person in people:
	for img, enc in person.images:
		known_face_encodings.append(enc)
		known_face_names.append(person.name)

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    gray = clahe.apply(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_gray = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_gray)
        face_encodings = face_recognition.face_encodings(rgb_gray, face_locations)

        face_names = []
        print(len(face_encodings))
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
                

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.flip(frame,-1)
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
