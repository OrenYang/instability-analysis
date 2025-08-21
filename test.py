import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
from scipy.interpolate import UnivariateSpline

PX_PER_MM = 10.0  # pixels per mm

class BoundaryAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Freehand Boundary Analyzer")

        self.canvas = tk.Canvas(root, cursor="cross", bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.left_points = []
        self.right_points = []
        self.drawing_side = "left"  # start with left boundary

        self.image = None
        self.tk_image = None

        self.instruction_label = tk.Label(root, text="Draw LEFT boundary (red), press Enter when done", fg="red")
        self.instruction_label.pack()

        self.load_image()

        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.root.bind("<Return>", self.next_boundary)

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select an image")
        if not file_path:
            self.root.destroy()
            return

        self.image = Image.open(file_path)
        self.tk_image = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor="nw")

    def on_click(self, event):
        self.add_point(event.x, event.y)

    def on_drag(self, event):
        self.add_point(event.x, event.y)

    def add_point(self, x, y):
        if self.drawing_side == "left":
            self.left_points.append((x, y))
            color = "red"
        else:
            self.right_points.append((x, y))
            color = "blue"

        self.canvas.create_oval(x-1, y-1, x+1, y+1, fill=color, outline=color)

    def next_boundary(self, event=None):
        if self.drawing_side == "left":
            if len(self.left_points) < 5:
                messagebox.showerror("Error", "Please draw more points for the left boundary.")
                return
            self.drawing_side = "right"
            self.instruction_label.config(text="Draw RIGHT boundary (blue), press Enter when done", fg="blue")
        else:
            if len(self.right_points) < 5:
                messagebox.showerror("Error", "Please draw more points for the right boundary.")
                return
            self.analyze_boundaries()

    def analyze_boundaries(self):
        left = np.array(self.left_points)
        right = np.array(self.right_points)

        # Sort by y
        left = left[np.argsort(left[:, 1])]
        right = right[np.argsort(right[:, 1])]

        # Remove duplicate y-values for spline fitting (keep first occurrence)
        def unique_rows(arr):
            _, idx = np.unique(arr[:,1], return_index=True)
            return arr[np.sort(idx)]

        left = unique_rows(left)
        right = unique_rows(right)

        # Common y-range inside overlapping region
        y_min = max(left[:, 1].min(), right[:, 1].min())
        y_max = min(left[:, 1].max(), right[:, 1].max())

        if y_max <= y_min:
            messagebox.showerror("Error", "No overlapping y-range between left and right boundaries.")
            return

        common_y = np.linspace(y_min, y_max, 200)

        try:
            # Use s=0 for interpolation (no smoothing)
            left_spline = UnivariateSpline(left[:, 1], left[:, 0], s=0)
            right_spline = UnivariateSpline(right[:, 1], right[:, 0], s=0)

            left_x_fit = left_spline(common_y)
            right_x_fit = right_spline(common_y)

            if np.any(np.isnan(left_x_fit)) or np.any(np.isnan(right_x_fit)):
                messagebox.showerror("Error", "NaN encountered in spline output.")
                return

        except Exception as e:
            messagebox.showerror("Error", f"Error during spline fitting:\n{e}")
            return

        half_width_mm = (right_x_fit - left_x_fit) / 2 / PX_PER_MM
        heights_mm = common_y / PX_PER_MM

        pinch_radius = np.mean(half_width_mm)
        instability = np.std(half_width_mm)

        # Show results
        messagebox.showinfo("Results",
                            f"Pinch radius: {pinch_radius:.3f} mm\n"
                            f"Instability: {instability:.3f} mm")

        # Draw smoothed curves on canvas
        for i in range(len(common_y) - 1):
            self.canvas.create_line(left_x_fit[i], common_y[i],
                                    left_x_fit[i+1], common_y[i+1],
                                    fill="orange", width=2)
            self.canvas.create_line(right_x_fit[i], common_y[i],
                                    right_x_fit[i+1], common_y[i+1],
                                    fill="cyan", width=2)

if __name__ == "__main__":
    root = tk.Tk()
    app = BoundaryAnalyzer(root)
    root.mainloop()
