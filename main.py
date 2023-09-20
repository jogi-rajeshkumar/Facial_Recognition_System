import tkinter as tk
from tkinter import ttk
import tkinter.messagebox
from tkinter import *
import cv2
import dlib
import random
from PIL import Image, ImageTk
import face_recognition,cv2,io,base64,os,pickle
import numpy as np
from numpy import expand_dims
from keras.preprocessing.image import load_img,img_to_array,ImageDataGenerator
from random import randrange



global keys
keys = {}
separator = "="
keys = {}
with open('config.properties') as f:
    for line in f:
        if separator in line:
            name, value = line.split(separator, 1)
            keys[name.strip()] = value.strip()

def show_jittered_images(window, jittered_images):

    for img in jittered_images:
        window.set_image(img)
        
datagen = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=[0.2,1.0],
    featurewise_center=True,
    featurewise_std_normalization=True)


class EmployeeRegistration:
    def __init__(self, root):
        self.root = root
        self.root.title("Registration")
        self.captured_photo_label = ttk.Label(root)
        self.captured_photo_label.grid(row=0, column=15, columnspan=6, pady=20)

        # self.root.attributes('-fullscreen', True)  # Make the window full screen
        
        width = 1920
        height = 1080
        root.geometry(f"{width}x{height}")

        self.employee_name = tk.StringVar()
        self.employee_id = tk.StringVar()
        self.organization = tk.StringVar()

        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 300)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

        self.label = ttk.Label(root)
        self.label.grid(row=0, column=0, columnspan=3, pady=20)

        style = ttk.Style()
        style.configure('TLabel', font=('Helvetica', 24))
        style.configure('TButton', font=('Helvetica', 18))
        style.configure('TEntry', font=('Helvetica', 18))

        self.name_label = ttk.Label(root, text="Name:")
        self.name_label.grid(row=1, column=0, sticky='w')
        self.name_entry = ttk.Entry(root, textvariable=self.employee_name)
        self.name_entry.grid(row=1, column=1, columnspan=2, pady=(0, 10))

        self.id_label = ttk.Label(root, text="Email_id/Phone No:")
        self.id_label.grid(row=2, column=0, sticky='w')
        self.id_entry = ttk.Entry(root, textvariable=self.employee_id)
        self.id_entry.grid(row=2, column=1, columnspan=2, pady=(0, 10))

        self.org_label = ttk.Label(root, text="Purpose:")
        self.org_label.grid(row=3, column=0, sticky='w')
        self.org_entry = ttk.Entry(root, textvariable=self.organization)
        self.org_entry.grid(row=3, column=1, columnspan=2, pady=(0, 10))

        self.capture_button = ttk.Button(root, text="Capture Photo", command=self.capture_photo)
        self.capture_button.grid(row=4, column=1, pady=20)

        self.retake_button = ttk.Button(root, text="Retake Photo", command=self.retake_photo, state=tk.DISABLED)
        self.retake_button.grid(row=4, column=2, pady=10)

        self.register_button = ttk.Button(root, text="Register", command=self.register_employee)
        self.register_button.grid(row=5, column=1, columnspan=2, pady=20)

        self.captured_photo = None
        self.update_camera()
        
        
    def update_camera(self):
        ret, frame = self.camera.read()
        if ret:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.label.config(image=self.photo)
            self.label.after(10, self.update_camera)


    def capture_photo(self):
        ret, frame = self.camera.read()
        if ret:
            self.captured_photo = frame
            #cv2.imshow('capture_photo', self.captured_photo)
            #cv2.waitKey(1000)
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.label.config(image=self.photo)
            self.capture_button.config(state=tk.DISABLED)
            self.retake_button.config(state=tk.NORMAL)
            self.display_captured_photo()
            
    '''def display_captured_photo(self):
        if self.captured_photo is not None:
            photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.captured_photo, cv2.COLOR_BGR2RGB)))
            self.label.config(image=photo)
            self.label.photo = photo'''
    def display_captured_photo(self):
        if self.captured_photo is not None:
            captured_image = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.captured_photo, cv2.COLOR_BGR2RGB)))
            self.captured_photo_label.config(image=captured_image)
            self.captured_photo_label.image = captured_image   
            
    
            
    # def show_captured_photo(self):
    #     if self.captured_photo is not None:
    #         captured_window = tk.Toplevel(self.root)
    #         captured_window.title("Captured Photo")
    #         captured_label = tk.Label(captured_window)
    #         captured_label.pack()
    #         captured_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(self.captured_photo, cv2.COLOR_BGR2RGB)))
    #         captured_label.config(image=captured_photo)
    #         captured_window.mainloop()

    def retake_photo(self):
        self.captured_photo = None
        self.label.config(image="")
        self.capture_button.config(state=tk.NORMAL)
        self.retake_button.config(state=tk.DISABLED)
        

    
    


    def register_employee(self):
        name = self.employee_name.get()
        emp_id = self.employee_id.get()
        org = self.organization.get()

        if name and emp_id and org and self.captured_photo is not None:
            photo_filename = f"{emp_id}.jpg"
            cv2.imwrite(photo_filename, self.captured_photo)
            print('.................',photo_filename)

            img = cv2.cvtColor(self.captured_photo, cv2.COLOR_BGR2RGB)
            #face_locations = face_recognition.face_locations(rgb,model = "hog")
            knownEncodings = []
            knownNames = []
            knownusernames = []
            # if image_dullness_score<=50 and image_blurness_score>=5:
            #if len(face_locations) != 0:
            predictor_path = '/home/kotesh/Music/shape_predictor_68_face_landmarks.dat'
            detector = dlib.get_frontal_face_detector()
            sp = dlib.shape_predictor(predictor_path)
            #img = dlib.load_rgb_image(face_locations)
            dets = detector(img)
            num_faces = len(dets)
            if num_faces is not None:
                faces = dlib.full_object_detections()
            
                for detection in dets:
                    faces.append(sp(img, detection))
  
                    #print(len(faces))
                    image = dlib.get_face_chip(img, faces[0], size=320)
                    #window = dlib.image_window()
                    #window.set_image(image)
                    random_number = randrange(2,5)
                    #print(random_number)
                    jittered_images = dlib.jitter_image(image, num_jitters=random_number, disturb_colors=True)
                    samples = expand_dims(image, 0)
                    it = datagen.flow(samples,batch_size=1)
                    image_data_images = [it.next()[0].astype('uint8') for i in range(random_number)]
                    for img in image_data_images:
                      jittered_images.append(img)
                      jittered_images.append(image)
                
                    
                    encodings = [face_recognition.face_encodings(image) for image in jittered_images]
                    #print("encodings<<<<<<<<<<<<<<<<<<",encodings)
                    for encoding in encodings:
                       #print('.........................',encoding)
                        knownEncodings.append(encoding[0])
                        knownNames.append(id)
                        knownusernames.append(name)
                    #print("[INFO] serializing encodings...")
                    encoding_file_path = keys['encodings']
                    if os.path.exists(encoding_file_path):
                        encoding_file_data = pickle.loads(open(keys['encodings'], "rb").read())
                        #print('the prevoius data is---------------------------------------------->',encoding_file_data)
                        for new_encodings,new_user_id in zip(knownEncodings,knownNames):
                            encoding_file_data["encodings"].append(new_encodings)
                            encoding_file_data["names"].append(new_user_id)
                            #print('the updated data is--------------------->',encoding_file_data)
                            f = open(keys['encodings'], "wb")
                            f.write(pickle.dumps(encoding_file_data))
                            f.close()
                    else:
                        encoding_file_data = {"encodings": knownEncodings, "names": knownNames}
                        #print(encoding_file_data)
                        f = open(keys['encodings'], "wb")
                        f.write(pickle.dumps(encoding_file_data))
                        f.close()
                    encoding_name_path = keys['encodings_name']
                    if os.path.exists(encoding_name_path):
                        encoding_name_data = pickle.loads(open(keys['encodings_name'], "rb").read())
                        #print('the prevoius data is---------------------------------------------->',encoding_file_data)
                        for new_usernames,new_user_id in zip(knownusernames,knownNames):
                            encoding_name_data["usernames"].append(new_usernames)
                            encoding_name_data["names"].append(new_user_id)
                            #print('the updated data is--------------------->',encoding_file_data)
                            f = open(keys['encodings_name'], "wb")
                            f.write(pickle.dumps(encoding_name_data))
                            f.close()
                    else:
                        encoding_name_data = {"usernames": knownusernames, "names": knownNames}
                        #print(encoding_file_data)
                        f = open(keys['encodings_name'], "wb")
                        f.write(pickle.dumps(encoding_name_data))
                        f.close()
                        
                #cv2.imwrite(photo_filename, self.captured_photo)
                self.display_captured_photo()
                print(f"Employee {name} registered with ID {emp_id} from {org}. Photo saved as {photo_filename}")
        # else:
        #     tkinter.messagebox.showinfo("Face Not Detected")
            
        else:
            print("Please provide all details and capture a photo.")
            


class VideoCapture:
    def __init__(self, root):
        self.root = root
        self.root.title("Webcam with Employee Data")
        self.root.geometry("1920x1080")

        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        webcam_frame = tk.Frame(main_frame)
        webcam_frame.pack(side=tk.LEFT, padx=10, pady=10, expand=True, fill=tk.BOTH)

        employee_frame = tk.Frame(main_frame, width=200)
        employee_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        self.webcam_label = tk.Label(webcam_frame)
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 300)
        self.cap.set(4, 300)

        self.update_webcam_feed()

        employee_label = ttk.Label(employee_frame, text="Employee Data")
        employee_label.pack(pady=10)

        detected_data = "Employee Name: Goutham\nDepartment: Developer"
        self.detected_label = tk.Label(employee_frame, text=detected_data, anchor="w", justify="left")
        self.detected_label.pack(fill=tk.X)

    def update_webcam_feed(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.resize(frame, (1600, 1200))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=img)
            self.webcam_label.img = img
            self.webcam_label.config(image=img)
            self.webcam_label.after(10, self.update_webcam_feed)

    def stop_capture(self):
        if hasattr(self, 'cap'):
            self.cap.release()
            self.root.quit()

class MainMenu:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial")
        self.root.geometry("1920x1080")  # Set the window size to 1920x1080

        # Create a label for the title
        title_label = tk.Label(root, text="Menu", font=("Helvetica", 40))
        title_label.pack(pady=100)

        # Create a frame for the buttons with a 3D effect
        button_frame = ttk.Frame(root, relief="raised", borderwidth=3)
        button_frame.pack(pady=150)

        # Define button styles
        button_style = ttk.Style()
        button_style.configure(
            "Custom.TButton",
            font=("Helvetica", 24),
            padding=20,
            background="#4CAF50",
            foreground="black",  # Set text color to black
        )
        button_style.map("Custom.TButton", background=[("active", "#45a049")])

        # Create the Verification button with custom style
        verification_button = ttk.Button(button_frame, text="Verification", command=self.verification_click, style="Custom.TButton")
        verification_button.pack(side="left", padx=50)

        # Create the Registration button with custom style
        registration_button = ttk.Button(button_frame, text="Registration", command=self.registration_click, style="Custom.TButton")
        registration_button.pack(side="right", padx=50)

    def verification_click(self):
        print("Verification button clicked")
        verification_window = tk.Toplevel(self.root)
        verification_window.title("Verification")
        video_capture = VideoCapture(verification_window)
    def registration_click(self):
        print("Registration button clicked")
        registration_window = tk.Toplevel(self.root)
        registration_gui = EmployeeRegistration(registration_window)

if __name__ == "__main__":
    root = tk.Tk()
    app = MainMenu(root)
    root.mainloop()

