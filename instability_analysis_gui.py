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
from joblib import Parallel, delayed

def autocorrelation_length(data):
    data = data - np.mean(data)
    n = len(data)
    acf = np.correlate(data, data, mode='full')[n-1:] / np.correlate(data, data, mode='full')[n-1]
    # Find where acf drops below 1/e
    try:
        lag_cutoff = np.where(acf < 1/np.e)[0][0]
    except IndexError:
        lag_cutoff = n  # no drop below threshold found
    return max(lag_cutoff, 1)  # avoid zero division


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
                  point_mode='all', N=5, forbidden_zones=None, draw_forbidden_zones=True,
                  title=None, total_points=None, resolution=None):
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
    if total_points is None:
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
    else:
        mode = f"{point_mode}{N}" if point_mode in ['inner', 'outer'] else point_mode

        sampled_ys = np.linspace(y_min, y_max, y_max - y_min + 1, dtype=int)  # All rows
        np.random.shuffle(sampled_ys)  # Randomize if you want more even sampling

        for y in sampled_ys:
            if len(left_x) >= total_points and len(right_x) >= total_points:
                break  # Stop once target is reached

            if y in left_points_by_y and left_points_by_y[y] and len(left_x) < total_points:
                selected = select_points(left_points_by_y[y], mode, 'left', peak_idx[y])
                selected = [x for x in selected if not in_forbidden_zone(x, y, forbidden_zones)]
                for x in selected:
                    if len(left_x) >= total_points:
                        break
                    left_x.append(x)
                    left_y.append(y)

            if y in right_points_by_y and right_points_by_y[y] and len(right_x) < total_points:
                selected = select_points(right_points_by_y[y], mode, 'right', peak_idx[y])
                selected = [x for x in selected if not in_forbidden_zone(x, y, forbidden_zones)]
                for x in selected:
                    if len(right_x) >= total_points:
                        break
                    right_x.append(x)
                    right_y.append(y)
        # Sort left and right points by y
        left_points = sorted(zip(left_y, left_x))
        right_points = sorted(zip(right_y, right_x))

        left_y, left_x = zip(*left_points) if left_points else ([], [])
        right_y, right_x = zip(*right_points) if right_points else ([], [])

        left_x, left_y = list(left_x), list(left_y)
        right_x, right_y = list(right_x), list(right_y)


    image_color = cv2.cvtColor(im_array, cv2.COLOR_GRAY2RGB)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image_color)

    if left_x:
        ax.plot(left_x, left_y, 'bo', markersize=1)
    if right_x:
        ax.plot(right_x, right_y, 'ro', markersize=1)

    pinch_radius = instability = left_angle = right_angle = avg_angle = None
    left_std = right_std = instability_se = None
    left_iqr = right_iqr = instability_iqr = instability_iqr_se = None
    angle_std = None
    left_mean = right_mean = None
    left_mrti = right_mrti = mrti_instability = mrti_instability_se = None

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
        left_res = np.array(left_x) - left_avg
        right_res = np.array(right_x) - right_avg

        left_ac_len = autocorrelation_length(left_res)
        right_ac_len = autocorrelation_length(right_res)

        left_N_eff = len(left_x) / left_ac_len
        right_N_eff = len(right_x) / right_ac_len

        left_sem = np.std(left_x) / np.sqrt(left_N_eff) / pxmm
        right_sem = np.std(right_x) / np.sqrt(right_N_eff) / pxmm
        radius_sem = np.sqrt(left_sem**2 + right_sem**2) / 2

        left_std = np.std(left_x) / pxmm
        right_std = np.std(right_x) / pxmm
        instability = (left_std + right_std) / 2

        left_iqr = (np.percentile(left_x, 75) - np.percentile(left_x, 25)) / pxmm
        right_iqr = (np.percentile(right_x, 75) - np.percentile(right_x, 25)) / pxmm
        instability_iqr = (left_iqr + right_iqr) / 2

        left_slope = left_coef[0]
        right_slope = right_coef[0]
        left_angle = np.degrees(np.arctan(abs(left_slope)))
        right_angle = np.degrees(np.arctan(abs(right_slope)))
        avg_angle = (left_angle + right_angle) / 2
        angle_std = np.std([left_angle, right_angle])

        left_mrti = np.std(left_res / pxmm)
        right_mrti = np.std(right_res / pxmm)
        mrti_instability = (left_mrti + right_mrti) / 2

        n_boot = 500  # adjust for speed/accuracy trade-off

        left_x = np.array(left_x)
        right_x = np.array(right_x)
        left_y = np.array(left_y)
        right_y = np.array(right_y)

        # === VECTORIZED BOOTSTRAP FOR STD INSTABILITY ===
        left_samples = np.random.choice(left_x, size=(n_boot, len(left_x)), replace=True)
        right_samples = np.random.choice(right_x, size=(n_boot, len(right_x)), replace=True)

        left_std_samples = np.std(left_samples, axis=1) / pxmm
        right_std_samples = np.std(right_samples, axis=1) / pxmm
        instability_boot = (left_std_samples + right_std_samples) / 2
        instability_se = np.std(instability_boot)

        # === VECTORIZED BOOTSTRAP FOR IQR INSTABILITY ===
        left_iqr_samples = (np.percentile(left_samples, 75, axis=1) - np.percentile(left_samples, 25, axis=1)) / pxmm
        right_iqr_samples = (np.percentile(right_samples, 75, axis=1) - np.percentile(right_samples, 25, axis=1)) / pxmm
        iqr_boot = (left_iqr_samples + right_iqr_samples) / 2
        instability_iqr_se = np.std(iqr_boot)

        # === PARALLEL BOOTSTRAP FOR MRTI INSTABILITY ===
        def bootstrap_mrti():
            li = np.random.choice(len(left_x), size=len(left_x), replace=True)
            ri = np.random.choice(len(right_x), size=len(right_x), replace=True)

            lx_b, ly_b = left_x[li], left_y[li]
            rx_b, ry_b = right_x[ri], right_y[ri]

            if np.ptp(ly_b) < 1e-5 or np.ptp(ry_b) < 1e-5:  # ptp = max-min, check if y variation is too small
                return None  # skip this bootstrap iteration

            lc = np.polyfit(ly_b, lx_b, 1)
            rc = np.polyfit(ry_b, rx_b, 1)

            left_res_b = lx_b - np.poly1d(lc)(ly_b)
            right_res_b = rx_b - np.poly1d(rc)(ry_b)

            l_mrti = np.std(left_res_b / pxmm)
            r_mrti = np.std(right_res_b / pxmm)
            return (l_mrti + r_mrti) / 2

        boot_mrti = Parallel(n_jobs=-1)(delayed(bootstrap_mrti)() for _ in range(n_boot))
        boot_mrti = [x for x in boot_mrti if x is not None]  # remove failed samples
        mrti_instability_se = np.std(boot_mrti)

    if resolution:
        radius_sem = np.sqrt(radius_sem**2+resolution**2)
        mrti_instability_se = np.sqrt(mrti_instability_se**2+resolution**2)
        instability_se = np.sqrt(instability_se**2+resolution**2)
        instability_iqr_se = np.sqrt(instability_iqr_se**2+resolution**2)


    if draw_forbidden_zones:
        for zx, zy, zw, zh in forbidden_zones:
            ax.add_patch(plt.Rectangle((zx, zy), zw, zh, color='red', alpha=0.2))

    if title:
        ax.set_title(title, fontsize=10)
    ax.axis('off')

    return (fig, pinch_radius, radius_sem, left_std, right_std, instability,
        instability_se, left_angle, right_angle, avg_angle, angle_std,
        left_mean, right_mean, left_iqr, right_iqr, instability_iqr,
        instability_iqr_se, left_mrti, right_mrti, mrti_instability,
        mrti_instability_se, len(left_x), len(right_x), left_x, left_y, right_x, right_y)


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

        self.total_points = Scale(controls, from_=0, to=4000, orient=HORIZONTAL, label="Total Points", command=self.on_slider_change)
        self.total_points.set(0)
        self.total_points.pack(anchor='w', pady=2)

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

        Label(controls, text="Resolution (mm):").pack(anchor='w')
        self.resolution_entry = Entry(controls)
        self.resolution_entry.insert(0, "0")
        self.resolution_entry.pack(anchor='w', pady=2)
        self.resolution_entry.bind("<FocusOut>", lambda event: self.update_plot())

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
        try:
            pinch_height = float(pinch_height_str)
        except (ValueError, TypeError):
            pinch_height = 13.5
        total_points = self.total_points.get() if self.total_points.get()!=0 else None
        resolution_str = self.resolution_entry.get()
        try:
            resolution = float(resolution_str)
        except (ValueError, TypeError):
            resolution = None

        new_fig, pinch_radius, radius_sem, left_instability, right_instability, \
        instability, instability_se, left_angle, right_angle, avg_angle, angle_std, \
        left_mean, right_mean, left_iqr, right_iqr, instability_iqr, instability_iqr_se, \
        left_mrti, right_mrti, mrti_instability, mrti_instability_se, left_points, right_points, \
        left_x, left_y, right_x, right_y = analyze_image(
            image_path, margin_top, margin_bot, threshold_fraction,
            pinch_height=pinch_height, point_mode=point_mode, N=N,
            forbidden_zones=self.forbidden_zones,
            draw_forbidden_zones=draw_forbidden_zones,
            title=image_name,
            total_points = total_points,
            resolution = resolution,
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
            'radius_sem': radius_sem,
            'left_instability': left_instability,
            'right_instability': right_instability,
            'instability': instability,
            'left_mrti': left_mrti,
            'right_mrti': right_mrti,
            'mrti_instability': mrti_instability,
            'mrti_instability_se': mrti_instability_se,
            'instability_se': instability_se,
            'left_iqr': left_iqr,
            'right_iqr': right_iqr,
            'instability_iqr': instability_iqr,
            'instability_iqr_se': instability_iqr_se,
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
            'pinch_height': pinch_height,
            'resolution': resolution,
            'total_points': total_points,
            'left_points': left_points,
            'right_points': right_points,
            'left_x': left_x,
            'left_y': left_y,
            'right_x': right_x,
            'right_y': right_y,
        }


        # Display
        result_text = (
            f"Pinch Radius: {fmt(pinch_radius, ' mm')} ± {fmt(radius_sem, ' mm')}\n"
            #f"Left MRTI Instability Amplitude: {fmt(left_mrti, ' mm')}\n"
            #f"Right MRTI Instability Amplitude: {fmt(right_mrti, ' mm')}\n"
            f"Avg. Instability Amplitude: {fmt(instability, ' mm')} ± {fmt(instability_se, ' mm')}\n"
            f"Avg. MRTI Instability: {fmt(mrti_instability, ' mm')} ± {fmt(mrti_instability_se, ' mm')}\n"
            f"Avg. Instability Amplitude - IQR: {fmt(instability_iqr, ' mm')} ± {fmt(instability_iqr_se, ' mm')}\n"
            #f"Left Flaring Angle: {fmt(left_angle, '°')}\n"
            #f"Right Flaring Angle: {fmt(right_angle, '°')}\n"
            f"Avg. Flaring Angle: {fmt(avg_angle, '°')} ± {fmt(angle_std, '°')}\n"
        )


        result_text += f"Timing: {fmt(timing, ' ns')}" if timing is not None else "Timing: N/A"

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
            #messagebox.showinfo("Figure Saved", f"Figure saved to:\n{output_path}")
            print("Figure Saved", f"Figure saved to:\n{output_path}")

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

                # Save boundary points as .npz
                fits_folder = os.path.join(self.output_folder_path, "fits")
                os.makedirs(fits_folder, exist_ok=True)

                base_name = os.path.splitext(filename)[0]
                npz_path = os.path.join(fits_folder, f"{base_name}_boundary_points.npz")

                np.savez(npz_path,
                         left_x=np.array(result.get('left_x', [])),
                         left_y=np.array(result.get('left_y', [])),
                         right_x=np.array(result.get('right_x', [])),
                         right_y=np.array(result.get('right_y', [])))


                # Create a row to append
                row = {
                    'Image': filename,
                    'Pinch Radius (mm)': result['pinch_radius'],
                    'Radius SEM (mm)': result['radius_sem'],
                    'Left Instability Amplitude (mm)': result['left_instability'],
                    'Right Instability Amplitude (mm)': result['right_instability'],
                    'Avg Instability Amplitude (mm)': result['instability'],
                    'Instability Amplitude SE (mm)': result['instability_se'],
                    'Left Instability MRTI (mm)': result['left_mrti'],
                    'Right Instability MRTI (mm)': result['right_mrti'],
                    'MRTI Instability (mm)': result['mrti_instability'],
                    'MRTI Instability SE (mm)': result['mrti_instability_se'],
                    'Left Instability Amplitude IQR (mm)': result['left_iqr'],
                    'Right Instability Amplitude IQR (mm)': result['right_iqr'],
                    'Avg Instability Amplitude IQR (mm)': result['instability_iqr'],
                    'Instability Amplitude SE IQR (mm)': result['instability_iqr_se'],
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
                    'Pinch Height (mm)': result['pinch_height'],
                    'Resolution (mm)': result['resolution'],
                    'Total Points': result['total_points'] if result['total_points'] is not None else "",
                    'Points in left boundary':result['left_points'],
                    'Points in right boundary':result['right_points']
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



def fmt(val, unit="", precision=2):
    return f"{val:.{precision}f}{unit}" if val is not None else "N/A"


if __name__ == "__main__":
    root = Tk()
    app = EdgeGUI(root)
    root.mainloop()
