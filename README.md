
# Pose Detection and Visualization using MediaPipe

This repository provides a Python script to detect human poses in an image using the MediaPipe library. The detected pose is visualized both in 2D (on the input image) and in 3D (using matplotlib). It also calculates a vector from the elbow to the wrist, computes the intersection of this vector with a defined plane, and visualizes the vector and intersection in the 3D plot.

## Requirements

The script requires the following Python libraries:
* MediaPipe
* OpenCV
* NumPy
* Matplotlib
* SymPy

The MediaPipe library can be installed via pip:

pip install mediapipe  

[Mediapipe](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python)  

[Look for more mediapipe tutorials](https://github.com/googlesamples/mediapipe)  


## Usage

To use the script, follow these steps:

1. Download the pose detection model:

bash
wget -O pose_landmarker.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

2. Download the input image:

bash
wget -q -O image.png https://img001.prntscr.com/file/img001/YODEqin4Sd63z33yOvTlww.png

3. Run the Python script.

## Functionality

The script executes the following steps:

1. It creates a PoseLandmarker object using the downloaded pose detection model.

2. It loads the input image.
![test image](https://i.postimg.cc/gcCdNjSP/16.jpg)

3. It uses the PoseLandmarker to detect the pose landmarks in the input image.

4. It visualizes the detected pose landmarks on the input image.
```
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
```
![Example of image](https://i.postimg.cc/0N3nw1S7/read.png)
5. It extracts the 3D coordinates of the pose landmarks and plots them in a 3D plot using matplotlib.
```
def WorldLandmarks_diagram(x, y, z):
  #Drawing up a diagram based on the obtained landmarks
  ax.scatter(x, y, z, color = 'skyblue')

  for c in POSE_CONNECTIONS:
    ax.plot([x[c[0]], x[c[1]]], [y[c[0]], y[c[1]]], [z[c[0]], z[c[1]]], color = 'skyblue')
```

6. It calculates a vector from the elbow to the wrist and visualizes it in the 3D plot.
```
def Vector(x, y, z):
  # vector construction from elbow to wrist
  point_elbow = np.array([x[13], y[13], z[13]])
  point_wrist = np.array([x[15], y[15], z[15]])
  vector = point_wrist - point_elbow
  print("Required vector is" + str(vector))

  ax.quiver(point_elbow[0], point_elbow[1], point_elbow[2], vector[0], vector[1], vector[2], color = 'navy')
  return vector
```

7. It defines a plane and calculates the intersection of the vector with the plane. The plane and the intersection point are also visualized in the 3D plot.
```
def Plane(x, y, z):
  # It was necessary to find a point close enough to the elbow so that there was no false triggering when the driver was leaning on the window but not extending his arm
  grid  = np.array([[1.2 * x[13], -1, 0],[1.2 * x[13], 1, 0],[1.2 * x[13], 1, 1],[1.2 * x[13] , -1, 1]])

  ax.add_collection3d(Poly3DCollection([grid], facecolor = 'g', alpha = 0.3))

  a = ['A', 'B', 'C', 'D']
  for i, j in enumerate(a):
    ax.text(grid[i][0], grid[i][1], grid[i][2], j)
  return grid
```
![alt text](https://i.postimg.cc/DztQzdVg/read2.png)
## Customization

The script can be customized by changing the parameters of the PoseLandmarkerOptions object in the script. For example, the confidence thresholds for pose detection, pose presence, and tracking can be adjusted to meet your specific needs.

The visualization of the pose landmarks can also be customized by changing the parameters of the draw_landmarks and scatter functions in the script.
