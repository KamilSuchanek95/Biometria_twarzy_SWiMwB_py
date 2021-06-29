
from gui_helper import *
import cv2
from functools import partial
import tkinter as tki
from functools import partial
from tkinter import messagebox

class NavBar(tki.Frame):


    def __init__(self, container, controller):
        super().__init__(container)
        self.controller = controller

        self.root = tki.Button(self, text="Recognition panel", command=partial(self.controller.show_frame, 0))
        self.root.pack(side='left')

        self.configuration = tki.Button(self, text="Configuration", command=partial(self.controller.show_frame, 1))
        self.configuration.pack(side='left')