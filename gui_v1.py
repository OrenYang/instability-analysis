import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Scale, Button, Label, HORIZONTAL, filedialog, StringVar, Frame, OptionMenu, Entry
from PIL import Image
from collections import defaultdict
import re
from matplotlib.path import Path

# Function to select edge points
def select_points(x_list, mode, side, center):
    x_sorted = sorted(x_list)
    if mode == 'all':
        return x_sorted

    match = re.match(r'(inner|outer)(\d+)', mode)
    if not match:
        raise ValueError("Invalid point_mode. Use 'all', 'innerN', or 'outerN'.")

    mode_type, n_str = match.groups()
    n = int(n_str)

    if len(x_list) <= n:
        return x_sorted

    if mode_type == 'inner':
        return sorted(x_list, key=lambda x: abs(x - center))[:n]
    elif mode_type == 'outer':
        half = n // 2
        remainder = n % 2

        if side == 'left':
            return x_sorted[:half + remainder]  # Left side points
        elif side == 'right':
            return x_sorted[-(half + remainder):]  # Right side points

# Check if a point is in any forbidden zone
def in_forbidden_zone(x, y, zones):
    for zx, zy, zw, zh in zones:
        if zx <= x <= zx + zw and zy <= y <= zy + zh:
            return True
    return False


# Modified analyze_image with forbidden_zones
def analyze_image(image_path, margin_top, margin_bot, threshold_fraction, pinch_height=13.5,
                  point_mode='all', N=5, forbidden_zones=None):
    if forbidden_zones is None:
        forbidden_zones = []

    analysis_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(analysis_image, (5, 5), 0)

    hgt, wid = blurred.shape
    min_intensity = np.min(blurred)
    peak_intensity = np.max(blurred)
    threshold_value = int(min_intensity + threshold_fraction * (peak_intensity - min_intensity))
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im_array = np.array(Image.open(image_path).convert('L'))
    center_x = wid // 2
    search_range = 35

    peak_idx = []
    for i in range(hgt):
        profile = im_array[i]
        left_bound = max(center_x - search_range, 0)
        right_bound = min(center_x + search_range, wid)
        local_max_index = np.argmax(profile[left_bound:right_bound]) + left_bound
        peak_idx.append(local_max_index)

    y_min, y_max = margin_top, hgt - margin_bot
    left_points_by_y = defaultdict(list)
    right_points_by_y = defaultdict(list)

    for contour in contours:
        for point in contour:
            x, y = point[0]
            if y_min <= y <= y_max and y < len(peak_idx):
                if x < peak_idx[y]:
                    left_points_by_y[y].append(x)
                elif x > peak_idx[y]:
                    right_points_by_y[y].append(x)

    left_x, left_y, right_x, right_y = [], [], [], []
    for y in range(y_min, y_max + 1):
        mode = f"{point_mode}{N}" if point_mode in ['inner', 'outer'] else point_mode

        if y in left_points_by_y:
            selected = select_points(left_points_by_y[y], mode, 'left', peak_idx[y])
            selected = [x for x in selected if not in_forbidden_zone(x, y, forbidden_zones)]
            left_x.extend(selected)
            left_y.extend([y] * len(selected))

        if y in right_points_by_y:
            selected = select_points(right_points_by_y[y], mode, 'right', peak_idx[y])
            selected = [x for x in selected if not in_forbidden_zone(x, y, forbidden_zones)]
            right_x.extend(selected)
            right_y.extend([y] * len(selected))

    image_color = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_color)

    if left_x: ax.plot(left_x, left_y, 'bo', markersize=0.05)
    if right_x: ax.plot(right_x, right_y, 'ro', markersize=0.05)

    pinch_radius = instability = left_angle = right_angle = avg_angle = None

    if left_x and right_x:
        left_coef = np.polyfit(left_y, left_x, 1)
        right_coef = np.polyfit(right_y, right_x, 1)
        left_avg = np.poly1d(left_coef)(left_y)
        right_avg = np.poly1d(right_coef)(right_y)
        ax.plot(left_avg, left_y, 'm--')
        ax.plot(right_avg, right_y, 'm--')

        left_mean = np.mean(left_x)
        right_mean = np.mean(right_x)
        ax.vlines([left_mean, right_mean], ymin=0, ymax=hgt, color='yellow', linestyle='dashed')

        pxmm = hgt / pinch_height
        pinch_radius = (right_mean - left_mean) / 2 / pxmm
        left_std = np.std(left_x)
        right_std = np.std(right_x)
        instability = (left_std + right_std) / 2 / pxmm

        left_slope = left_coef[0]
        right_slope = right_coef[0]
        left_angle = np.degrees(np.arctan(abs(left_slope)))
        right_angle = np.degrees(np.arctan(abs(right_slope)))
        avg_angle = (left_angle + right_angle) / 2

    # Draw forbidden zones
    for zx, zy, zw, zh in forbidden_zones:
        ax.add_patch(plt.Rectangle((zx, zy), zw, zh, color='red', alpha=0.2))

    ax.axis('off')
    return fig, pinch_radius, instability, left_angle, right_angle, avg_angle


# GUI Class
class EdgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Analysis GUI")
        self.image_path = None
        self.forbidden_zones = []
        self.rect_patch = None
        self.start_x = self.start_y = None
        self.dragging = False

        # Split layout into two frames: left (plot) and right (controls)
        main_frame = Frame(root)
        main_frame.pack(fill='both', expand=True)

        self.plot_frame = Frame(main_frame)
        self.plot_frame.pack(side='left', fill='both', expand=True)

        controls = Frame(main_frame)
        controls.pack(side='right', fill='y', padx=10, pady=10)

        # Control widgets
        self.load_button = Button(controls, text="Load Image", command=self.load_image)
        self.load_button.pack(anchor='w', pady=2)

        self.margin_top_scale = Scale(controls, from_=0, to=100, orient=HORIZONTAL, label="Top Margin", command=self.on_slider_change)
        self.margin_top_scale.set(15)
        self.margin_top_scale.pack(anchor='w', pady=2)

        self.margin_bot_scale = Scale(controls, from_=0, to=100, orient=HORIZONTAL, label="Bottom Margin", command=self.on_slider_change)
        self.margin_bot_scale.set(15)
        self.margin_bot_scale.pack(anchor='w', pady=2)

        self.threshold_scale = Scale(controls, from_=0, to=100, orient=HORIZONTAL, label="Threshold (%)", command=self.on_slider_change)
        self.threshold_scale.set(40)
        self.threshold_scale.pack(anchor='w', pady=2)

        self.point_mode_var = StringVar()
        self.point_mode_var.set('all')
        self.point_mode_menu = OptionMenu(controls, self.point_mode_var, 'all', 'inner', 'outer')
        self.point_mode_menu.pack(anchor='w', pady=2)
        self.point_mode_var.trace_add("write", lambda *args: self.update_plot())

        self.N_slider = Scale(controls, from_=1, to=20, orient=HORIZONTAL, label="N for Inner/Outer", command=self.on_slider_change)
        self.N_slider.set(5)
        self.N_slider.pack(anchor='w', pady=2)

        Label(controls, text="Pinch Height (mm):").pack(anchor='w')
        self.pinches_height_entry = Entry(controls)
        self.pinches_height_entry.insert(0, "13.5")
        self.pinches_height_entry.pack(anchor='w', pady=2)
        self.pinches_height_entry.bind("<FocusOut>", lambda event: self.update_plot())

        self.clear_button = Button(controls, text="Clear Zones", command=self.clear_zones)
        self.clear_button.pack(anchor='w', pady=5)

        self.result_label = Label(controls, text="Results will appear here.", justify='left', anchor='w', font=("Arial", 10))
        self.result_label.pack(anchor='w', padx=5, pady=10)

        # Initial matplotlib figure
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        # Event bindings
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)


    def on_slider_change(self, event=None):
        self.update_plot()

    def load_image(self):
        file_path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.png *.jpg *.tif")])
        if file_path:
            self.image_path = file_path
            self.forbidden_zones.clear()
            self.update_plot()

    def update_plot(self):
        if not self.image_path:
            return
        margin_top = self.margin_top_scale.get()
        margin_bot = self.margin_bot_scale.get()
        threshold_fraction = self.threshold_scale.get() / 100
        point_mode = self.point_mode_var.get()
        N = self.N_slider.get()
        pinch_height_str = self.pinches_height_entry.get()
        pinch_height = float(pinch_height_str) if pinch_height_str else 13.5

        new_fig, pinch_radius, instability, left_angle, right_angle, avg_angle = analyze_image(
            self.image_path, margin_top, margin_bot, threshold_fraction,
            pinch_height=pinch_height, point_mode=point_mode, N=N,
            forbidden_zones=self.forbidden_zones
        )

        self.canvas.get_tk_widget().pack_forget()
        self.fig = new_fig
        self.ax = self.fig.axes[0]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        result_text = (
            f"Pinch Radius: {pinch_radius:.2f} mm\n"
            f"Instability Amplitude: {instability:.2f} mm\n"
            f"Left Flare Angle: {left_angle:.2f}°\n"
            f"Right Flare Angle: {right_angle:.2f}°\n"
            f"Avg. Flare Angle: {avg_angle:.2f}°"
        )
        self.result_label.config(text=result_text)

    def on_mouse_press(self, event):
        if event.inaxes:
            self.dragging = True
            self.start_x, self.start_y = event.xdata, event.ydata
            self.rect_patch = plt.Rectangle((self.start_x, self.start_y), 0, 0,
                                            linewidth=1, edgecolor='r', facecolor='r', alpha=0.3)
            self.ax.add_patch(self.rect_patch)
            self.canvas.draw()

    def on_mouse_drag(self, event):
        if self.dragging and event.inaxes and self.rect_patch:
            width = event.xdata - self.start_x
            height = event.ydata - self.start_y
            self.rect_patch.set_width(width)
            self.rect_patch.set_height(height)
            self.canvas.draw()

    def on_mouse_release(self, event):
        if self.dragging and event.inaxes:
            self.dragging = False
            end_x, end_y = event.xdata, event.ydata
            rect = (
                min(self.start_x, end_x),
                min(self.start_y, end_y),
                abs(end_x - self.start_x),
                abs(end_y - self.start_y)
            )
            self.forbidden_zones.append(rect)
            self.update_plot()

    def clear_zones(self):
        self.forbidden_zones.clear()
        self.update_plot()

# Main block
if __name__ == "__main__":
    root = Tk()
    app = EdgeGUI(root)
    root.mainloop()
