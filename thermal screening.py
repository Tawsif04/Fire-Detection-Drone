
import numpy as np
import cv2

threshold = 240
area_of_box = 700
min_temp = 20
font_scale_caution = 1
font_scale_temp = 0.7


def convert_to_temperature(pixel_avg):
    """Converts pixel value (mean) to temperature """
    a= pixel_avg/0.08
    celsius = (a-32)*5/9
    return celsius


def process_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    heatmap_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    heatmap = cv2.applyColorMap(heatmap_gray, cv2.COLORMAP_HOT)

    # Binary threshold
    _, binary_thresh = cv2.threshold(heatmap_gray, threshold, 255, cv2.THRESH_BINARY)

    # Image opening: Erosion followed by dilation. We do this to decrease the white noise. But this reduces object size.
    # The image is dilated again to increase the size but noise won't return

    kernel = np.ones((3, 3), np.uint8)
    image_erosion = cv2.erode(binary_thresh, kernel, iterations=1)
    image_opening = cv2.dilate(image_erosion, kernel, iterations=1)

    # Get contours from the image obtained by opening operation
    # each contour is a numpy array
    contours, _ = cv2.findContours(image_opening, 1, 2)

    image_with_rectangles = np.copy(heatmap)

    for contour in contours:
        # rectangle over each contour
        x,y,w,h = cv2.boundingRect(contour)

        # Pass if the area of rectangle is not large enough
        if (w)*(h) < area_of_box:
            continue

        # Mask is boolean type of matrix.
        mask = np.zeros_like(heatmap_gray)
        cv2.drawContours(mask, contour, -1, 255, -1)

        # Mean of only those pixels which are in blocks and not the whole rectangle selected
        mean = convert_to_temperature(cv2.mean(heatmap_gray, mask=mask)[0])

        # Colors for rectangles and text min_area
        temperature = round(mean, 2)
        color = (0, 255, 0) if temperature < min_temp else (
            255, 255, 127)

        # Callback function if the following condition is true
        if temperature >= min_temp:
            # Call back function here
            cv2.putText(image_with_rectangles, "FIRE DETECTED", (35, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale_caution, color, 2, cv2.LINE_AA)

        # Draw rectangles for visualisation
        image_with_rectangles = cv2.rectangle(
            image_with_rectangles, (x, y), (x+w, y+h), color, 2)

        # Write temperature for each rectangle
        cv2.putText(image_with_rectangles, "{} degrees C".format(temperature), (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale_temp, color, 2, cv2.LINE_AA)

    return image_with_rectangles

def main():
    """Main driver function"""
    # For Video Input
    # video = cv2.VideoCapture("F:\\3-1\\Project Lab\\Thermal_Screening_Temperature_Detection-main\\production ID_3769037.mp4")
    video=cv2.VideoCapture('http://172.20.10.3:8080//video')
   
    video_frames = []
    while True:
        ret, frame = video.read()

        if not ret:
            break

        # Process each frame
        frame = process_frame(frame)
        height, width, _ = frame.shape
        video_frames.append(frame)
        resized=cv2.resize(frame,(800,600))
        # Show the video as it is being processed in a window
        cv2.imshow('frame', resized)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    video.release()
    cv2.destroyAllWindows()

    # Save video to output
    #size = (height, width)
    #out = cv2.VideoWriter(str(base_dir+'output.mp4'),cv2.VideoWriter_fourcc(*'MJPG'), 100, size)

    #for i in range(len(video_frames)):
        #out.write(video_frames[i])
    #out.release()


if __name__ == "__main__":
    main()
