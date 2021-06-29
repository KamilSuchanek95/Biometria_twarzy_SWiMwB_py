from program.helper.gui_helper import *
from program.tools.detector_and_recognitor import *
from program.frames.configuration_frame import ConfigurationFrame
from program.frames.recognition_frame import RecognitionFrame
from program.frames.nav_bar import NavBar

import tkinter as tki


class RecognitionApp(tki.Tk):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        create_resources_if_they_dont_exists()

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
        
        self.recognizer = None
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

