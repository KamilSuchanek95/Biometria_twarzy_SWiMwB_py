#!/usr/bin/python3

from gui_helper import *
from configuration_frame import ConfigurationFrame
from recognition_frame import RecognitionFrame
from nav_bar import NavBar
from detector_and_recognitor import *


import tkinter as tki


class RecognitionApp(tki.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wm_title("Prototyp aplikacji systemu biometrycznego")
        self.config(background="#004890")
        self.protocol("WM_DELETE_WINDOW", self.on_close) 

        container = tki.Frame(self)
        container.pack(fill=tki.BOTH, expand=True)

        self.navbar = NavBar(container, self)
        self.navbar.grid(row=0, column=0, sticky='nsew')

        self.frames = {}
        self.frames[0] = RecognitionFrame(container, self)
        self.frames[0].grid(row=1, column=0, sticky="nsew")
        self.frames[1] = ConfigurationFrame(container, self)
        self.frames[1].grid(row=1, column=0, sticky="nsew")
        
        self.show_frame(0)
        
        # globals
        self.detector = Face_detector()
        self.recognizer = Face_recognitor()
        self.identities = {}
        self.eukli_distances = {}

        self.frames[1].check_if_program_is_ready()

    def show_frame(self, frame_num):
        for frame in self.frames.values():
            frame.grid_remove()
        frame = self.frames[frame_num]
        frame.grid()

    def on_close(self):
        print("[INFO] closing...")
        self.frames[0].turn_off_camera()
        self.quit()
        self.destroy()

    
   




def start_application():
    create_resources_if_they_dont_exists()
    appli = RecognitionApp()
    appli.mainloop()


start_application()

