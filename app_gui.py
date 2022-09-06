from LPRPredict import *
from io import BytesIO
import cv2
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def read_image_file(file):
    image = Image.open(file)
    image = np.array(image)[:,:,::-1]
    return image

LPRPredictor = LicensePlateRecognition()

#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('License Plate Recognition')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

def classify(file_path):
    image = read_image_file(file_path)
    license_plate = LPRPredictor.predict(image)
    print(license_plate)
    label.configure(foreground='#011638', text=license_plate)

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",
                        command=lambda: classify(file_path),
                        padx=10,pady=5)

    classify_b.configure(background='#364156',
                        foreground='white',
                        font=('arial',10,'bold'))

    classify_b.place(relx=0.79,rely=0.46)

def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

upload = Button(top, text="UPLOAD AN IMAGE", command=upload_image, padx=10, pady=5)
upload.configure(background='#9cb5e6', foreground='white', font=('arial', 10, 'bold'))
upload.pack(side=BOTTOM, pady=50)
sign_image.pack(side=BOTTOM, expand=True)
label.pack(side=BOTTOM, expand=True)
heading = Label(top, text="License Plate Recognition",
                pady=20, font=('arial', 20, 'bold'))
heading.configure(background='#fccaca', foreground='#336bd6')
heading.pack()
top.mainloop()

