import cv2  # Importing the OpenCV library
import os  # Importing the os library to interact with the file system


# Function to extract frames
def extract_frames(video_path, output_folder, frame_rate=0.5):
    # Check if output_folder does not exist and create it
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the video
    video = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video.isOpened():
        raise ValueError("Error opening video file")

    # Get the video's FPS (Frames Per Second)
    fps = video.get(cv2.CAP_PROP_FPS)

    # Calculate the interval between frames we want to save
    interval = int(fps * frame_rate)

    # Initialize frame count
    count = 0

    # Loop through frames
    while True:
        # Read the next frame from the video
        success, frame = video.read()

        # If read was not successful, the video has ended
        if not success:
            break

        # Save frame every 'interval' frames
        if count % interval == 0:
            # Construct filename
            filename = os.path.join(output_folder, f"frameee_{count // interval:05d}.jpg")
            # Write the frame to the file
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")

        # Increase frame count
        count += 1

    # Release the video capture object
    video.release()
    print("Finished extracting frames.")


# Example usage
video_path = 'IMG_1477.MOV'  # Replace with your video path
output_folder = 'img'  # Replace with your desired output path

# Uncomment the line below to run the function with the example paths
extract_frames(video_path, output_folder)