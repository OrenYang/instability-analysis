Code to fit outline of pinch and determine pinch radius and instability amplitude

RUN instability_analysis_gui.py TO OPEN APP FOR ANALYZING IMAGES. UPLOAD IMAGES OR FOLDER OF IMAGES CROPPED AND NAMED AS DESCRIBED BELOW. UPLOAD TIMING INPUT, EXAMPLE FORMAT IS LISTED BELOW. this is the most useful file, all others can be ignored. mess with the parameters in this gui to create a good fit and then update the csv and save the image. click and drag on the image to create "forbidden regions" if it is detecting random points.

instability_analysis.py is the mian file to interact with if done more manually. It calls on contour_analysis.py and timings.py. 

canny_analysis.py and gradient_analysis.py attempt to do the same thing as contour analysis but were less succesful and not used in the final instability_analysis code.

To start the analysis, crop the images so there is only one frame per image and the height of the images matches the pinch height. This can be done in imagej or any other image editting software.
The cropped image files need to be renamed "XXXX_MCPX_X" for example "2500_MCP1_3" where the last digit is the frame number. All images should be saved into one folder.

The timings.py code can get timings either directly from shotsheets (timings.shotsheet()) or through a custom excel file (timings.excel()). the excel file should have the format:


Shot             1234     1235    1236     1237    1238

mcp1 frame 1     201.6    310.2   134.0    240.4   235

mcp1 frame 2     205.6    216.5   178.4    140.3   168

mcp1 frame 3     221.1    310.9   234.2    220.4   120

mcp1 frame 4     101.6    345.8   134.7    140.9   270

mcp2 frame 1     251.5    210.2   134.0    140.4   171

mcp2 frame 2     141.8    110.1   209.9    390.8   174

mcp2 frame 3     291.8    320.2   134.0    150.5   180

mcp2 frame 4     301.6    340.2   217.4    120.4   160

Where the value in the table is the timing of the frame.

Within the instability_analysis.py code, you will adjust some parameters for each image/image folder. 
"margin_top" and "margin_bot" determine how many pixels are ignored from the ends of the pinch. This is done to ignore any unreal contours that would effect pinch radius and instability
"threshold_fraction" determines where the pinch boundary is drawn and is a value between 0 and 1. If it is close to 1 the boundary will be closer to the highest intensity, close to zero will extend the boundary past the edge of the pinch.
"Save" is a boolean that determines whether the fitted images will be saved to the output folder.
"image_dir" is the path to the directory with the cropped images
"pinch_height" is the height of the pinch in mm, this is used to convert px to mm, since the image height should match the pinch height.

margin_top and margin_bot may vary between images in a folder, so if variable is set to True these can be adjusted every shot. The code will show the image and then prompt for a new threshold and margins.
It will then display the new image and ask if any more changes are needed. It will continue the cycle until all images have been satisfactorily fit.

The output of the code will save a csv with shot number, camera number, frame number, timing, pinch radius, and instability amplitude called output.csv. It will save to a folder called "output".
if an output folder already exsists it will save to "output1"... The analyzed images will also save to this folder with their original name + "_fit".
