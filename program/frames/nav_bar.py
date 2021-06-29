import tkinter as tki
from functools import partial

class NavBar(tki.Frame):


    def __init__(self, container, controller):
        super().__init__(container)

        self.root = tki.Button(self, text="Recognition panel", command=partial(controller.show_frame, 0))
        self.root.pack(side='left')

        self.configuration = tki.Button(self, text="Configuration", command=partial(controller.show_frame, 1))
        self.configuration.pack(side='left')