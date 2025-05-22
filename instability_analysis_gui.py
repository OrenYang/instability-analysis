import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import Tk, Scale, Button, Label, HORIZONTAL, filedialog, StringVar, Frame, OptionMenu, Entry, messagebox
from PIL import Image
from collections import defaultdict
import re
import os
from matplotlib.path import Path
import csv
import pandas as pd

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

# Image analysis function
def analyze_image(image_path, margin_top, margin_bot, threshold_fraction, pinch_height=13.5,
                  point_mode='all', N=5, forbidden_zones=None, draw_forbidden_zones=True, title=None):
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

    if left_x: ax.plot(left_x, left_y, 'bo', markersize=1)
    if right_x: ax.plot(right_x, right_y, 'ro', markersize=1)

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
        left_std = np.std(left_x) / pxmm
        right_std = np.std(right_x) / pxmm
        instability = (left_std + right_std) / 2
        instability_std = np.std([left_std, right_std])

        left_slope = left_coef[0]
        right_slope = right_coef[0]
        left_angle = np.degrees(np.arctan(abs(left_slope)))
        right_angle = np.degrees(np.arctan(abs(right_slope)))
        avg_angle = (left_angle + right_angle) / 2
        angle_std = np.std([left_angle, right_angle])

    if draw_forbidden_zones:
        for zx, zy, zw, zh in forbidden_zones:
            ax.add_patch(plt.Rectangle((zx, zy), zw, zh, color='red', alpha=0.2))

    if title:
        ax.set_title(title, fontsize=10)
    ax.axis('off')

    return fig, pinch_radius, left_std, right_std, instability, instability_std, left_angle, right_angle, avg_angle, angle_std, left_mean, right_mean


# GUI Class
class EdgeGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Edge Analysis GUI")
        self.image_list = []
        self.current_index = -1
        self.forbidden_zones = []
        self.rect_patch = None
        self.start_x = self.start_y = None
        self.dragging = False
        self.output_folder_path = False
        self.timing_df = None
        self.image_settings = {}  # Store results per image
        self.current_image = None


        main_frame = Frame(root)
        main_frame.pack(fill='both', expand=True)

        self.plot_frame = Frame(main_frame)
        self.plot_frame.pack(side='left', fill='both', expand=True)

        controls = Frame(main_frame)
        controls.pack(side='right', fill='y', padx=10, pady=10)

        # Load buttons
        self.load_image_button = Button(controls, text="Load Image", command=self.load_single_image)
        self.load_image_button.pack(anchor='w', pady=2)

        self.load_folder_button = Button(controls, text="Load Folder", command=self.load_folder)
        self.load_folder_button.pack(anchor='w', pady=2)

        timing_frame = Frame(controls)
        timing_frame.pack(anchor='w', pady=2)

        self.load_timing_button = Button(timing_frame, text="Load Timing File", command=self.load_timing_file)
        self.load_timing_button.pack(side='left')

        help_label = Label(timing_frame, text="ℹ️", fg="blue", cursor="question_arrow", font=("Arial", 12))
        help_label.pack(side='left', padx=5)
        help_label.bind("<Button-1>", self.show_timing_help)

        # Previous/Next buttons
        nav_frame = Frame(controls)
        nav_frame.pack(anchor='w', pady=2)

        self.prev_button = Button(nav_frame, text="Previous", command=self.show_previous_image)
        self.prev_button.pack(side='left', padx=(0, 5))

        self.next_button = Button(nav_frame, text="Next", command=self.show_next_image)
        self.next_button.pack(side='left')

        # Remaining controls (unchanged)
        self.margin_top_scale = Scale(controls, from_=0, to=300, orient=HORIZONTAL, label="Top Margin", command=self.on_slider_change)
        self.margin_top_scale.set(15)
        self.margin_top_scale.pack(anchor='w', pady=2)

        self.margin_bot_scale = Scale(controls, from_=0, to=300, orient=HORIZONTAL, label="Bottom Margin", command=self.on_slider_change)
        self.margin_bot_scale.set(15)
        self.margin_bot_scale.pack(anchor='w', pady=2)

        self.threshold_scale = Scale(controls, from_=0, to=100, orient=HORIZONTAL, label="Threshold (%)", command=self.on_slider_change)
        self.threshold_scale.set(40)
        self.threshold_scale.pack(anchor='w', pady=2)

        point_mode_frame = Frame(controls)
        point_mode_frame.pack(anchor='w', pady=2)

        Label(point_mode_frame, text="Mode:").pack(side='left', padx=(0, 5))

        self.point_mode_var = StringVar()
        self.point_mode_var.set('all')
        self.point_mode_menu = OptionMenu(point_mode_frame, self.point_mode_var, 'all', 'inner', 'outer')
        self.point_mode_menu.pack(side='left')
        self.point_mode_var.trace_add("write", lambda *args: self.update_plot())

        # N value label (replaces "N:")
        self.N_value_label = Label(point_mode_frame, text=str(5))
        self.N_value_label.pack(side='left', padx=(10, 5))

        # Slider without number on top
        self.N_slider = Scale(
            point_mode_frame, from_=1, to=20, orient=HORIZONTAL,
            length=100, showvalue=False, command=self.on_slider_change
        )
        self.N_slider.set(5)
        self.N_slider.pack(side='left')

        # Update label with slider value
        def update_N_label(val):
            self.N_value_label.config(text=str(int(float(val))))
            self.on_slider_change(val)

        self.N_slider.config(command=update_N_label)

        Label(controls, text="Pinch Height (mm):").pack(anchor='w')
        self.pinches_height_entry = Entry(controls)
        self.pinches_height_entry.insert(0, "13.5")
        self.pinches_height_entry.pack(anchor='w', pady=2)
        self.pinches_height_entry.bind("<FocusOut>", lambda event: self.update_plot())

        self.clear_button = Button(controls, text="Clear Zones", command=self.clear_zones)
        self.clear_button.pack(anchor='w', pady=5)

        #buttons for saving
        self.output_folder_button = Button(controls, text="Select Output Folder", command=self.load_output_folder)
        self.output_folder_button.pack(anchor='w', pady=2)

        self.output_path_label = Label(controls, text="No output folder selected", justify='left', anchor='w', wraplength=200, font=("Arial", 10))
        self.output_path_label.pack(anchor='w', padx=5, pady=10)

        self.save_image_button = Button(controls, text="Save Image", command=self.save_image)
        self.save_image_button.pack(anchor='w', pady=5)

        self.update_csv_button = Button(controls, text="Update CSV", command=lambda: self.write_results_to_csv())
        self.update_csv_button.pack(anchor='w', pady=5)

        self.result_label = Label(controls, text="Results will appear here.", justify='left', anchor='w', font=("Arial", 10))
        self.result_label.pack(anchor='w', padx=5, pady=10)

        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def load_single_image(self):
        file_path = filedialog.askopenfilename(title="Select image", filetypes=[("Image files", "*.png *.jpg *.tif")])
        if file_path:
            self.image_list.append(file_path)
            self.current_index = len(self.image_list) - 1
            self.forbidden_zones.clear()
            self.update_plot()

    def load_folder(self):
        folder_path = filedialog.askdirectory(title="Select folder")
        if folder_path:
            extensions = (".png", ".jpg", ".jpeg", ".tif")
            files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(extensions)]
            files.sort()
            self.image_list.extend(files)
            if self.current_index == -1 and self.image_list:
                self.current_index = 0
            self.forbidden_zones.clear()
            self.update_plot()

    def load_timing_file(self):
        file_path = filedialog.askopenfilename(title="Select timing file", filetypes=[("*.csv *.xlsx")])
        if file_path:
            df = pd.read_excel(file_path, header=None)

            # Adjust df labels to index by shot number and "mcpx frame x"
            df.columns = df.iloc[0]
            df = df[1:].reset_index(drop=True)
            df.set_index(df.columns[0], inplace=True)
            df.columns = df.columns.astype(int)
            self.timing_df = df

    def show_timing_help(self, event=None):
        help_text = (
            "Load a timing file (CSV or Excel) where:\n"
            "- The first row contains shot numbers.\n"
            "- The first column contains labels like 'mcpx frame x'.\n"
            "- The cell at (row='mcpx frame x', column=shot number) gives the timing in µs.\n\n"
            "Make sure the camera/frame names in image filenames match this format: 1234_MCP1_1.jpg, 1234_SCH_2.jpg, ..."
        )
        messagebox.showinfo("Timing File Help", help_text)

    def load_output_folder(self):
        self.output_folder_path = filedialog.askdirectory(title="Select folder")
        self.output_path_label.config(text=self.output_folder_path)

    def show_previous_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.forbidden_zones.clear()
            self.restore_settings()
            self.update_plot()

    def show_next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.forbidden_zones.clear()
            self.restore_settings()
            self.update_plot()

    def on_slider_change(self, event=None):
        self.update_plot()

    def update_plot(self, draw_forbidden_zones=True):
        if self.current_index == -1 or not self.image_list:
            return

        image_path = self.image_list[self.current_index]
        image_name = os.path.basename(image_path)
        self.current_image = image_path

        # Parse shot, cam, frame from filename
        try:
            shot = int(image_name[:4])
            cam = image_name.split('_')[1]
            frame = image_name.split('_')[2][0]
        except (ValueError, IndexError):
            print(f"Skipping {image_name}: filename format incorrect")
            timing = None
        else:
            if self.timing_df is not None and shot in self.timing_df.columns:
                try:
                    timing = self.timing_df.loc[f'{cam.lower()} frame {frame}', shot]
                except KeyError:
                    print(f"Timing column missing for {image_name}")
                    timing = None
            else:
                timing = None

        margin_top = self.margin_top_scale.get()
        margin_bot = self.margin_bot_scale.get()
        threshold_fraction = self.threshold_scale.get() / 100
        point_mode = self.point_mode_var.get()
        N = self.N_slider.get()
        pinch_height_str = self.pinches_height_entry.get()
        pinch_height = float(pinch_height_str) if pinch_height_str else 13.5

        new_fig, pinch_radius, left_instability, right_instability, instability, instability_std, left_angle, right_angle, avg_angle, angle_std, left_mean, right_mean = analyze_image(
            image_path, margin_top, margin_bot, threshold_fraction,
            pinch_height=pinch_height, point_mode=point_mode, N=N,
            forbidden_zones=self.forbidden_zones,
            draw_forbidden_zones=draw_forbidden_zones,
            title=image_name
        )

        # Update plot
        self.canvas.get_tk_widget().pack_forget()
        self.fig = new_fig
        self.ax = self.fig.axes[0]
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side='bottom', fill='both', expand=True)
        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_drag)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

        # Store results
        self.image_settings[image_path] = {
            'pinch_radius': pinch_radius,
            'left_instability': left_instability,
            'right_instability': right_instability,
            'instability': instability,
            'instability_std': instability_std,
            'left_angle': left_angle,
            'right_angle': right_angle,
            'avg_angle': avg_angle,
            'angle_std': angle_std,
            'timing': timing,
            'margin_top': margin_top,
            'margin_bot': margin_bot,
            'threshold_fraction': threshold_fraction,
            'point_mode': point_mode,
            'N': N,
            'pinch_height': pinch_height
        }


        # Display
        result_text = (
            f"Pinch Radius: {pinch_radius:.2f} mm\n"
            f"Left Instability Amplitude: {left_instability:.2f} mm\n"
            f"Right Instability Amplitude: {right_instability:.2f} mm\n"
            f"Avg. Instability Amplitude: {instability:.2f} mm\n"
            f"Instability Amplitude Std.: {instability_std:.2f} mm\n"
            f"Left Flaring Angle: {left_angle:.2f}°\n"
            f"Right Flaring Angle: {right_angle:.2f}°\n"
            f"Avg. Flaring Angle: {avg_angle:.2f}°\n"
            f"Flare Angle Std.: {angle_std:.2f}°\n"
        )
        if timing is not None:
            result_text += f"Timing: {timing:.2f} ns"
        else:
            result_text += "Timing: N/A"

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

    def restore_settings(self):
        image_path = self.image_list[self.current_index]
        if image_path not in self.image_settings:
            return  # Nothing to restore

        settings = self.image_settings[image_path]

        self.margin_top_scale.set(settings['margin_top'])
        self.margin_bot_scale.set(settings['margin_bot'])
        self.threshold_scale.set(int(settings['threshold_fraction'] * 100))
        self.point_mode_var.set(settings['point_mode'])
        self.N_slider.set(settings['N'])
        self.pinches_height_entry.delete(0, 'end')
        self.pinches_height_entry.insert(0, str(settings['pinch_height']))

    def save_image(self):
        # Save the figure (matplotlib plot) without drawing forbidden zones
        if self.current_index == -1 or not self.image_list:
            return

        output_folder = self.output_folder_path
        if output_folder:
            image_path = self.image_list[self.current_index]
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            output_path = os.path.join(output_folder, f"{base_name}_fit.png")

            # Update the plot without drawing forbidden zones
            self.update_plot(draw_forbidden_zones=False)

            # Save the figure
            self.fig.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Figure saved to {output_path}")

    def write_results_to_csv(self):
        if not self.output_folder_path:
            messagebox.showerror("Error", "No output folder selected.")
            return

        output_file = os.path.join(self.output_folder_path, "results.csv")

        # Load existing CSV if it exists, otherwise create a new one
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
        else:
            df = pd.DataFrame(columns=[
                'Image', 'Timing (ns)',
            ])

        # Prepare new rows
        rows = []
        for img_path, result in self.image_settings.items():
            if img_path == self.current_image:
                filename = os.path.basename(img_path)

                # Create a row to append
                row = {
                    'Image': filename,
                    'Pinch Radius (mm)': result['pinch_radius'],
                    'Left Instability Amplitude (mm)': result['left_instability'],
                    'Right Instability Amplitude (mm)': result['right_instability'],
                    'Avg Instability Amplitude (mm)': result['instability'],
                    'Instability Amplitude std (mm)': result['instability_std'],
                    'Left Flaring Angle (deg)': result['left_angle'],
                    'Right Flaring Angle (deg)': result['right_angle'],
                    'Avg Flaring Angle (deg)': result['avg_angle'],
                    'Flaring Angle std (deg)': result['angle_std'],
                    'Timing (ns)': result['timing'] if result['timing'] is not None else "",
                    'Top Margin': result['margin_top'],
                    'Bottom Margin': result['margin_bot'],
                    'Threshold (%)': result['threshold_fraction'] * 100,
                    'Point Mode': result['point_mode'],
                    'N': result['N'],
                    'Pinch Height (mm)': result['pinch_height']
                }

                # Check if this image already exists in the CSV, and update it if necessary
                if filename in df['Image'].values:
                    df.loc[df['Image'] == filename, list(row.keys())] = list(row.values())
                else:
                    rows.append(row)

        # If there are new rows, append them to the DataFrame
        if rows:
            new_rows_df = pd.DataFrame(rows)
            df = pd.concat([df, new_rows_df], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        df.to_csv(output_file, index=False)
        messagebox.showinfo("CSV Saved", f"Results saved to:\n{output_file}")





if __name__ == "__main__":
    root = Tk()
    app = EdgeGUI(root)
    root.mainloop()
