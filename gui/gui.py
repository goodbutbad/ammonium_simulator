import tkinter as tk
from tkinter import ttk, messagebox
from model.ammonium_model import AmmoniumSoilModel
from utils.plotting import create_figure
import threading
import numpy as np

class AmmoniumModelGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Модель NH4")
        self.root.geometry("1200x800")
        self.model = AmmoniumSoilModel()
        self.setup_gui()

    def setup_gui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='y')
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        self.setup_parameters(left_frame)
        self.setup_plots(right_frame)

    def setup_parameters(self, parent):
        canvas = tk.Canvas(parent, width=300)
        scrollbar = ttk.Scrollbar(parent, orient='vertical', command=canvas.yview)
        scrollable = ttk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)

        self.param_vars = {}
        phys = [("D", 5.0), ("v", 2.0), ("k_a", 0.05), ("k_d", 0.02),
                ("C_max", 50.0), ("k_nitr", 0.01), ("V_max", 2.0), ("K_m", 5.0)]
        for i, (k, v) in enumerate(phys):
            ttk.Label(scrollable, text=k).grid(row=i, column=0, sticky='w')
            var = tk.DoubleVar(value=v)
            self.param_vars[k] = var
            ttk.Entry(scrollable, textvariable=var).grid(row=i, column=1)

        sim = [("z_max", 50.0), ("t_max", 15.0), ("nz", 25), ("nt", 500)]
        for i, (k, v) in enumerate(sim, start=len(phys)):
            ttk.Label(scrollable, text=k).grid(row=i, column=0, sticky='w')
            var = tk.IntVar(value=v) if isinstance(v, int) else tk.DoubleVar(value=v)
            self.param_vars[k] = var
            ttk.Entry(scrollable, textvariable=var).grid(row=i, column=1)

        init = [("C_initial", 20.0), ("C_ads_initial", 5.0)]
        for i, (k, v) in enumerate(init, start=len(phys)+len(sim)):
            ttk.Label(scrollable, text=k).grid(row=i, column=0, sticky='w')
            var = tk.DoubleVar(value=v)
            self.param_vars[k] = var
            ttk.Entry(scrollable, textvariable=var).grid(row=i, column=1)

        ttk.Button(scrollable, text="Запустить", command=self.run_simulation).grid(pady=5)
        ttk.Button(scrollable, text="Сохранить", command=self.save_results).grid(pady=5)
        ttk.Button(scrollable, text="Сброс", command=self.reset).grid(pady=5)

        canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

    def setup_plots(self, parent):
        self.tabs = ttk.Notebook(parent)
        self.tabs.pack(fill='both', expand=True)
        self.depth_tab = ttk.Frame(self.tabs)
        self.time_tab = ttk.Frame(self.tabs)
        self.contour_tab = ttk.Frame(self.tabs)
        self.tabs.add(self.depth_tab, text='По глубине')
        self.tabs.add(self.time_tab, text='По времени')
        self.tabs.add(self.contour_tab, text='2D')
        self.fig1, self.ax1, self.canvas1 = create_figure(self.depth_tab, "", "", "")
        self.fig2, self.ax2, self.canvas2 = create_figure(self.time_tab, "", "", "")
        self.fig3, self.ax3, self.canvas3 = create_figure(self.contour_tab, "", "", "")

    def update_model(self):
        for k, var in self.param_vars.items():
            setattr(self.model, k, var.get())

    def run_simulation(self):
        self.update_model()
        w = tk.Toplevel(self.root)
        w.title("...")
        ttk.Label(w, text="Ждите...").pack(pady=20)
        bar = ttk.Progressbar(w, mode='indeterminate')
        bar.pack(pady=10)
        bar.start()
        def run():
            try:
                self.model.solve_model()
                self.root.after(0, lambda: self.simulation_done(w))
            except Exception as e:
                self.root.after(0, lambda: self.simulation_fail(w, str(e)))
        threading.Thread(target=run, daemon=True).start()

    def simulation_done(self, w):
        w.destroy()
        self.update_plots()
        messagebox.showinfo("Готово", "Моделирование завершено")

    def simulation_fail(self, w, msg):
        w.destroy()
        messagebox.showerror("Ошибка", msg)

    def update_plots(self):
        if self.model.C is None:
            return
        self.ax1.clear()
        times = [0, len(self.model.t)//4, len(self.model.t)//2, -1]
        for i in times:
            self.ax1.plot(self.model.C[i], self.model.z)
        self.ax1.invert_yaxis()
        self.canvas1.draw()
        self.ax2.clear()
        depths = [0, len(self.model.z)//4, len(self.model.z)//2, -1]
        for i in depths:
            self.ax2.plot(self.model.t, self.model.C[:, i])
        self.canvas2.draw()
        self.ax3.clear()
        T, Z = np.meshgrid(self.model.t, self.model.z)
        cs = self.ax3.contourf(T, Z, self.model.C.T, levels=20)
        self.fig3.colorbar(cs, ax=self.ax3)
        self.ax3.invert_yaxis()
        self.canvas3.draw()

    def save_results(self):
        if self.model.C is None:
            return
        try:
            np.savez('results/ammonium_model_results.npz',
                     C=self.model.C,
                     C_ads=self.model.C_ads,
                     z=self.model.z,
                     t=self.model.t)
            messagebox.showinfo("Сохранено", "Сохранено")
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def reset(self):
        defaults = {
            "D": 5.0, "v": 2.0, "k_a": 0.05, "k_d": 0.02, "C_max": 50.0,
            "k_nitr": 0.01, "V_max": 2.0, "K_m": 5.0, "z_max": 50.0,
            "t_max": 15.0, "nz": 25, "nt": 500, "C_initial": 20.0, "C_ads_initial": 5.0
        }
        for k, v in defaults.items():
            if k in self.param_vars:
                self.param_vars[k].set(v)
