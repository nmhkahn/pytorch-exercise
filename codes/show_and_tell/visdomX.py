import numpy as np
from visdom import Visdom

class VisdomX:
    def __init__(self):
        self.vis = Visdom()
        self.wins = dict()

    def add_scalars(self, 
                    value, step,
                    title="", 
                    ylabel="Loss", xlabel="Epoch"):
        # visdom only get np.array or tensor
        value = np.array([value])
        step = np.array([step], dtype=np.int64)

        if not self.wins.get(title):
            win = self.vis.line(
                Y=value, X=step,
                opts=dict(
                    title=title, 
                    ylabel=ylabel, 
                    xlabel=xlabel
                )
            )
            self.wins[title] = win
        else:
            self.vis.line(
                Y=value, X=step,
                win=self.wins.get(title), 
                update="append"
            )

    def add_text(self, text):
        self.vis.text(text)
