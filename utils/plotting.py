import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def create_figure(parent, title, xlabel, ylabel):
    fig = Figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    canvas = FigureCanvasTkAgg(fig, parent)
    canvas.get_tk_widget().pack(fill='both', expand=True)
    return fig, ax, canvas
