# -*- coding: utf-8 -*-
"""
Created on Wed Nov 18 09:56:14 2020

@author: Deni
"""
from tkinter import *
import tkinter as tk
import tkinter.messagebox
import os
import trainingProcess as train

root = tk.Tk()
root.title('Face Recognition HDD Apps')

canvas = tk.Canvas(root, height = 300, width = 400, bg = "#263D42")
canvas.pack()


#creating command
def openfile():
    path = "D:\Computer Vision\ReadImage\imagesAttandance"
    path = os.path.realpath(path)
    os.startfile(path)
    print(path)
    pass

def openCamera():
    name = train.openCamera()
    messageInfo = "Data kecatat atas nama " + str(name)
    tk.messagebox.showinfo("ShowInfo", messageInfo)

def trainingData():
    status = train.trainingProcess()
    messageInfo = "Training " + str(status)
    tk.messagebox.showinfo("ShowInfo", messageInfo)

def about():
    messageInfo = "Aplikasi face Recognition untuk mencatat data kehadiran mahasiswa di HDD"
    tk.messagebox.showinfo("ShowInfo", messageInfo)


#Starting menu
my_menu = Menu(root)
root.config(menu = my_menu)

#menu Item
file_menu = Menu(my_menu)
my_menu.add_cascade(label="Options", menu = file_menu)
file_menu.add_command(label="Open file Training", command = openfile)
file_menu.add_separator()
file_menu.add_command(label="About", command = about)

button = tk.Button(canvas, text = "Open Camera", command = openCamera)
button.place(relx = 0.3, rely = 0.15, relwidth = 0.4, relheight = 0.25)

button = tk.Button(canvas, text = "Training Image", command = trainingData)
button.place(relx = 0.3, rely = 0.55, relwidth = 0.4, relheight = 0.25)


root.mainloop()