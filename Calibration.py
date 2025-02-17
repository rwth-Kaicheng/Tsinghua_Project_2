import cv2
import numpy as np
import glob

def calibrate_camera(images_path, chessboard_size=(6, 9), square_size=1.0):
    """
    Calibrates the camera using a series of chessboard images to obtain the intrinsic matrix and distortion coefficients.

    Parameters:
    - images_path: File path pattern containing chessboard images
    - chessboard_size: Size of the chessboard (rows, columns), e.g., (6, 9).
    - square_size: The actual size of each chessboard square, in millimeters, centimeters, etc.

    Returns:
    - camera_matrix: Camera intrinsic matrix.
    - dist_coeffs: Distortion coefficients.
    - rvecs: Rotation vectors.
    - tvecs: Translation vectors.
    """
    # Prepare world coordinates for chessboard corners
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size

    # Lists to store 3D points in the world and 2D points in the image
    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    # Read all image files
    images = glob.glob(images_path)

    for fname in images[::10]:  # Process every 10th image to reduce computation
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (chessboard_size[0], chessboard_size[1]), None)

        # If enough corners are found, add them to the list
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display detected corners
            cv2.drawChessboardCorners(img, (chessboard_size[0], chessboard_size[1]), corners, ret)
            cv2.imshow('Chessboard Corners', img)
            cv2.waitKey(1)

        # Perform camera calibration
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    return camera_matrix, dist_coeffs, rvecs, tvecs

# Calibration
images_path = 'F:/rawdata_20240703/cam/cam1/*.bmp'
camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(images_path)

print("Camera Matrix:")
print(camera_matrix)
print("\nDistortion Coefficients:")
print(dist_coeffs)
