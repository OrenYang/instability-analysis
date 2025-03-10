import os
from contour_analysis import find_edges
import mcp_timings
import csv

########### PARAMETERS TO ADJUST #############
save = True # save analyzed figures (True/False)
image_dir = "cropped_images" # Cropped image directory, images must be cropped and rotated to match pinch height
margin_top = 70 #px     Amount of image not included in the analysis, in case of poorly traced edges
margin_bot = 10 #px
threshold_fraction = 0.25 # Fraction between max and min intensity to define boundary (0-1)
pinch_height = 13.5 #mm
variable = True # Change threshold and margins for every image in folder (True/False)

# Choose how to get timing information, mcp_timings.shotsheet() for directly from a UCSD shotsheet
# mcp_timings.excel() for custom excel file
# Both will open file dialog to choose file and return a dataframe
timing_df = mcp_timings.excel()
# Example of excel format:
# Shot              1234   1235    1236    1237
# mcp1 frame 1      100.6  123.7   205.4   202
# mcp1 frame 2      102.6  123.3   205.9   209
# mcp2 frame 1      143.5  124.7   207.0   212.1

##############################################



############# DONT CHANGE ####################
# create output folder
output_folder = 'output/'
counter = 1
while os.path.exists(output_folder):
    output_folder = f'output{counter}/'
    counter += 1
os.makedirs(output_folder)
# Write analyzed radii and instability amplitude to csv file
with open(f'{output_folder}output.csv', 'w', newline='') as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Shot','QC','Frame','Timing','Radius','Instability Amplitude'])

    # Loop through image directory analyzing each image
    for img in os.listdir(image_dir):
        bad = True # parameter determining if you need to adjust threshold multiple times
        path = os.path.join(image_dir, img)
        shot = float(img[:4])
        cam = img.split('_')[1]
        frame = img.split('_')[2][0]
        timing = timing_df[shot]['{} frame {}'.format(cam.lower(), frame)]

        # Optionally adjust threshold and margin each image
        if variable:
            find_edges(path, margin_top, margin_bot, threshold_fraction, pinch_height)
            while bad:
                threshold_fraction = float(input('Threshold Fraction: '))
                margin_top = int(input('Top margin (px): '))
                margin_bot = int(input('Bottom margin (px): '))
                radius, instability, _, _, _, _ = find_edges(path, margin_top, margin_bot, threshold_fraction, pinch_height, save, output_folder)
                bad = input('Change threshold? (y/n): ').lower().strip() == 'y'
        else:
            radius, instability, _, _, _, _ = find_edges(path, margin_top, margin_bot, threshold_fraction, pinch_height, save, output_folder)

        writer.writerow([int(shot), cam, frame, timing, radius, instability])
