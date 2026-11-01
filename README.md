# Bahraini Currency Recognition

## Objective
Develop a simple computer vision application that detects and recognizes Bahraini banknotes and coins in images using a custom-trained **YOLO26s** object detection model.

## Features
- Upload a currency image or take a picture with a camera.
- Detect currency and draw bounding boxes around the detected items.
- Display the currency denomination and confidence score directly on the image.

## Model Performance

Results from the **test set (44 images)**:

| Metric | Result |
|---|---|
| Precision | **96.7%** |
| Recall | **83.3%** |

**Note:** mAP@50 measures object detection performance, not classification accuracy. A separate classification accuracy score was not calculated.

## Technologies Used
- Python
- Ultralytics YOLO26s
- Streamlit
- Roboflow

## How to Run

1. Place the trained `best_2.pt` model in the same folder as `app.py`.
2. Install the requirements:

   `pip install -r requirements.txt`

3. Start the application:

   `streamlit run app.py`

## Limitations
- The test set was small (44 images) and did not include coin examples, so coin recognition performance has not been established.
- Detection may be affected by lighting, background, image quality, and camera angle.
- The dataset still needs cleanup for inconsistent annotations and the generic `currency` label.
