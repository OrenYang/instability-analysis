import os
from contour_analysis import find_edges
import timings
import csv

########### PARAMETERS TO ADJUST #############
save = True # save analyzed figures (True/False)
image_dir = "cropped_images_edited" # Cropped image directory, images must be cropped and rotated to match pinch height
original_image_dir = "cropped_images"
margin_top = 15 #px     Amount of image not included in the analysis, in case of poorly traced edges
margin_bot = 15 #px
threshold_fraction = 0.4 # Fraction between max and min intensity to define boundary (0-1)
pinch_height = 13.5 #mm
variable = True # Change threshold and margins for every image in folder (True/False)
point_mode = 'all' # limit the number of points taken at each y value to weight each position the same. use 'all', 'innerN', or 'outerN' where N is an integer

# Choose how to get timing information, mcp_timings.shotsheet() for directly from a UCSD shotsheet
# mcp_timings.excel() for custom excel file
# Both will open file dialog to choose file and return a dataframe
timing_df = timings.excel()
# Example of excel format:
# Shot              1234   1235    1236    1237
# mcp1 frame 1      100.6  123.7   205.4   202
# mcp1 frame 2      102.6  123.3   205.9   209
# mcp2 frame 1      143.5  124.7   207.0   212.1

##############################################



############# DONT CHANGE ####################
# create output folder
if save:
    # Create output folder
    output_folder = 'output/'
    counter = 1
    while os.path.exists(output_folder):
        output_folder = f'output{counter}/'
        counter += 1
    os.makedirs(output_folder)

    # Open CSV to write output
    output_csv_path = os.path.join(output_folder, 'output.csv')
    csvfile = open(output_csv_path, 'w', newline='')
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Shot','QC','Frame','Timing (ns)','Radius (mm)','Instability Amplitude (mm)','Left Angle (deg)','Right Angle (deg)','Average Angle (deg)'])

# Loop through image directory analyzing each image
for img in os.listdir(image_dir):
    bad = True # parameter determining if you need to adjust threshold multiple times
    path = os.path.join(image_dir, img)
    try:
        shot = float(img[:4])
        cam = img.split('_')[1]
        frame = img.split('_')[2][0]
    except (ValueError, IndexError):
        print(f"Skipping {img}: filename format incorrect")
        continue

    # Skip images that dont have shots in the timing df
    if shot not in timing_df:
        print(f"Skipping {img}: Shot {shot} not in timing data")
        continue

    timing = timing_df[shot]['{} frame {}'.format(cam.lower(), frame)]
    display_path = os.path.join(original_image_dir, img)

    # Optionally adjust threshold and margin each image
    if variable:
        find_edges(path, margin_top, margin_bot, threshold_fraction, pinch_height, point_mode=point_mode, display_image_path=display_path)
        while bad:
            threshold_fraction = float(input('Threshold Fraction: '))
            margin_top = int(input('Top margin (px): '))
            margin_bot = int(input('Bottom margin (px): '))
            point_mode = input('which points to use: ')
            radius, instability, _, _, _, _, left_angle, right_angle, avg_angle = find_edges(path, margin_top, margin_bot, threshold_fraction, pinch_height, save, output_folder if save else '', point_mode=point_mode, display_image_path = display_path)
            bad = input('Change threshold? (y/n): ').lower().strip() == 'y'
    else:
        radius, instability, *_ = find_edges(path, margin_top, margin_bot, threshold_fraction, pinch_height, save, output_folder if save else '', point_mode = point_mode, display_image_path = display_path)

    if save:
        writer.writerow([int(shot), cam, frame, timing, radius, instability, left_angle, right_angle, avg_angle])
if save:
    csvfile.close()
