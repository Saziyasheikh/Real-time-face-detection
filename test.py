import urllib.request

url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

urllib.request.urlretrieve(url, "face_detection.onnx")

print("Download Complete")