# YOLOv8 Vehicle Detection and Tracking on UA-DETRAC Dataset

This project demonstrates a pipeline for vehicle detection and tracking using YOLOv8 on the UA-DETRAC dataset. It covers dataset downloading, image preprocessing, video creation from image sequences, model training, inference on images and videos, and model evaluation.

## Overview

The workflow involves:
1.  **Dataset Acquisition**: Downloading the UA-DETRAC dataset from Kaggle.
2.  **Data Exploration & Preprocessing**: Loading images, resizing, normalization, and applying augmentations like random occlusions.
3.  **Video Preparation**: Creating an MP4 video from a sequence of training images, which can serve as a test video.
4.  **Model Training**: Training a YOLOv8 model on the UA-DETRAC dataset for vehicle detection.
5.  **Inference**:
    * Performing detection on single test images.
    * Performing detection and tracking on videos, drawing bounding boxes with class-specific colors, and displaying per-class object counts.
6.  **Evaluation**: Calculating mAP metrics to assess model performance.

## Features

* Automated dataset download from Kaggle.
* Standard image preprocessing techniques (resize, normalize).
* Example data augmentation (random occlusion).
* YOLOv8 model training with custom data.
* Inference on images and videos.
* Object tracking using ByteTrack with YOLOv8.
* Visualization of detection results with class-specific colors and unique track IDs.
* Per-class object counting in videos.
* Model performance evaluation (mAP).

## Dataset

This project uses the [UA-DETRAC dataset](https://www.kaggle.com/datasets/dtrnngc/ua-detrac-dataset), a challenging real-world dataset for multi-vehicle detection and tracking. The dataset is downloaded using the Kaggle API.

## Setup and Installation

### Prerequisites
* Python 3.x
* pip (Python package installer)
* Kaggle account and API token (`kaggle.json`)

### Installation Steps

1.  **Clone the repository (Example):**
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```

2.  **Install Dependencies:**
    The scripts require several Python libraries. You can install them using pip:
    ```bash
    pip install opencv-python-headless matplotlib albumentations ultralytics kaggle kagglehub -q
    ```
    * `opencv-python-headless`: For image and video processing.
    * `matplotlib`: For displaying images.
    * `albumentations`: For advanced image augmentation (optional, but included).
    * `ultralytics`: For YOLOv8 training and inference.
    * `kaggle`, `kagglehub`: For downloading the dataset.

3.  **Kaggle API Setup:**
    * Download your `kaggle.json` API token from your Kaggle account page.
    * The script `explore&preprocess.py` includes a cell to upload `kaggle.json` directly in a Colab environment. If running locally, ensure `kaggle.json` is in the expected location (e.g., `~/.kaggle/kaggle.json`).

4.  **Google Drive (Optional, for Colab users):**
    The provided scripts are designed for Google Colab and assume the dataset and some paths might be on Google Drive. Adjust paths accordingly if running in a different environment.
    * `dataset_base_path = "/content/drive/MyDrive/UA_DATA"`

## Workflow and Usage

The project is primarily divided into two parts, based on the provided Python scripts:
1.  `explore&preprocess.py`: Focuses on dataset loading, exploration, and basic preprocessing/augmentation.
2.  `train&process-2.py`: Focuses on creating a test video, training the YOLOv8 model, and performing advanced inference and evaluation.

### Part 1: Dataset Exploration and Preprocessing (from `explore&preprocess.py`)

This script handles the initial steps of acquiring and preparing the data.

1.  **Download UA-DETRAC Dataset:**
    The script uses `kagglehub.dataset_download("dtrnngc/ua-detrac-dataset")` to fetch the dataset. Ensure your Kaggle API token is set up.

2.  **Load and Display Sample Image:**
    A sample image from the training set is loaded and displayed to verify data access.
    * **Important**: You'll need to adjust `dataset_base_path` to where your dataset is stored. The script attempts to find images in `images/train` and labels in `labels/train` subdirectories.

3.  **Image Resizing:**
    Images are resized to a target dimension (e.g., 640x640), which is a common input size for YOLO models.
    ```python
    # Example from explore&preprocess.py
    target_size = (640, 640)
    resized_image = cv2.resize(image, target_size)
    ```

4.  **Image Normalization:**
    Pixel values are normalized from the 0-255 range to 0-1.
    ```python
    # Example from explore&preprocess.py
    normalized_image = resized_image.astype(np.float32) / 255.0
    ```

5.  **Random Occlusion (Augmentation Example):**
    Random black rectangles are drawn on the image to simulate occlusions, a form of data augmentation.

### Part 2: Video Creation, Model Training, and Inference (from `train&process-2.py` and `explore&preprocess.py`)

This part covers generating a test video, training the model, and using it for detection and tracking.

1.  **Create MP4 Video from Images (from `train&process-2.py`):**
    This section creates an MP4 video from a sequence of images (e.g., from the training set) to be used for testing the tracking capabilities.
    * **Important**: Modify `image_folder` to point to your directory of images.
    * Adjust `video_name`, `fps`, and `num_images_to_process` as needed.
    ```python
    # Key parameters in train&process-2.py
    image_folder = '/content/UA_DATA/images/train' # !! REPLACE THIS !!
    video_name = '/content/traffic_video_1000frames_24fps.mp4'
    fps = 24
    num_images_to_process = 1000
    ```

2.  **Train YOLOv8 Model (from `train&process-2.py`):**
    A YOLOv8 model is trained using the UA-DETRAC dataset.
    * **Configuration File**: Training requires a YAML file (e.g., `ua_detrac_config.yaml`) specifying paths to training/validation data and class names. Ensure this file is correctly set up.
    * **Base Model**: The script loads a base model (e.g., `yolov8n.pt` or `yolov8s.pt`). The provided script mentions `yolo11n.pt`; you should use a standard YOLOv8 model like `yolov8n.pt`.
    ```python
    # Example from train&process-2.py
    model = YOLO('yolov8n.pt') # Or 'yolov8s.pt', etc.
    results = model.train(
        data='/content/ua_detrac_config.yaml', # Path to your dataset config file
        epochs=50, # Number of training epochs
        imgsz=416, # Image size for training
        batch=128, # Batch size (adjust based on GPU memory)
        name='yolov8s_ua_detrac_run1' # Name for the training run directory
    )
    ```
    The trained model weights (e.g., `best.pt`) will be saved in a directory like `runs/detect/yolov8s_ua_detrac_run1/weights/`.

3.  **Inference on a Single Test Image (from `train&process-2.py`):**
    After training, you can test the model on individual images.
    * **Important**: Set `path_to_test_image_on_drive` to the path of your test image.
    * The script loads the trained model (make sure the path to `best.pt` is correct).
    * Detections are visualized and raw prediction data (class, confidence, bounding box) is printed.

4.  **Video Inference and Tracking (Combined logic from both scripts):**
    Both scripts contain sections for video inference. `explore&preprocess.py` shows basic video inference, while `train&process-2.py` demonstrates advanced tracking with per-class counting and colored bounding boxes.

    * **Load Trained Model:**
        ```python
        # Example from train&process-2.py
        path_to_your_trained_model = "/content/runs/detect/yolov8s_ua_detrac_run1/weights/best.pt" # Adjust this path
        model = YOLO(path_to_your_trained_model)
        ```

    * **Perform Tracking and Counting (from `train&process-2.py`):**
        This script performs object tracking (using ByteTrack by default) and counts unique objects per class.
        * **Important**:
            * Set `path_to_your_test_video` to the input video file (e.g., the one created in step 1).
            * `output_video_name_per_class` defines the name of the processed output video.
        * Detections are drawn with class-specific colors.
        * A running count of unique tracked objects for each class is displayed on the video frames.
        ```python
        # Key call from train&process-2.py
        results_generator = model.track(
            source=path_to_your_test_video, # Input video
            stream=True,
            persist=True, # Persist tracks between frames
            tracker='bytetrack.yaml', # Tracking algorithm
            conf=0.3,
            iou=0.5,
            save=False # Manual drawing and saving is implemented
        )
        ```

    * **Basic Video Inference (from `explore&preprocess.py`):**
        This script also shows a simpler way to process a video and save the output with detections.
        ```python
        # Key call from explore&preprocess.py
        # path_to_your_test_video should be defined
        results_generator = model(
            source=path_to_your_test_video, # Input video
            stream=True,
            save=True,   # Save the output video automatically by YOLO
            conf=0.3,
            iou=0.5,
            name="video_inference_run" # Output directory name
        )
        ```

5.  **Evaluate Model Accuracy (from `train&process-2.py`):**
    The script evaluates the trained model on a validation set (defined in your `ua_detrac_config.yaml`).
    * **Important**: Ensure the path to your trained model (`best.pt`) and the `ua_detrac_config.yaml` are correct. The original script had an error where it reloaded `yolov8n.pt` instead of the trained model for validation; this should be corrected to use your `best.pt`.
    ```python
    # Corrected example logic from train&process-2.py
    model = YOLO('/content/runs/detect/yolov8s_ua_detrac_run1/weights/best.pt') # Load your trained model
    metrics = model.val(data='/content/drive/MyDrive/ua_detrac_config.yaml') # Path to your dataset config file
    print(f"mAP50-95: {metrics.box.map}")    # mAP50-95
    print(f"mAP50: {metrics.box.map50}")  # mAP@.5
    print(f"mAP75: {metrics.box.map75}")  # mAP@.75
    # print(metrics.box.maps)   # List of mAPs
    # print(metrics.results_dict) # Detailed results per class
    ```

## Important Configuration Paths

Users **must** update these paths in the scripts to match their environment and file locations:

**From `explore&preprocess.py`:**
* `dataset_base_path`: Path to the base directory where the UA-DETRAC dataset is stored (e.g., `/content/drive/MyDrive/UA_DATA`).
* `path_to_your_trained_model`: Path to your trained `best.pt` file for video inference (e.g., `/content/runs/detect/yolov8s_ua_detrac_run1/weights/best.pt`).
* `path_to_your_test_video`: Path to the video file you want to process for inference (if not using the one generated by `train&process-2.py`).

**From `train&process-2.py`:**
* `image_folder`: Path to the directory containing images to create the MP4 video (e.g., `/content/UA_DATA/images/train`).
* `data` (in `model.train()`): Path to your `ua_detrac_config.yaml` file.
* `path_to_test_image_on_drive`: Path to a single image for testing the trained model (e.g., `/content/UA_DATA/images/train/MVI_20011_img00012.jpg`).
* `path_to_your_trained_model`: Path to your trained `best.pt` file for video processing and evaluation.
* `path_to_your_test_video`: Path to the input video for tracking and counting (e.g., `/content/traffic_video_1000frames_24fps.mp4`).
* `data` (in `model.val()`): Path to your `ua_detrac_config.yaml` file for evaluation.

## Dependencies
* `opencv-python-headless`
* `numpy`
* `matplotlib`
* `os` (standard library)
* `random` (standard library)
* `albumentations`
* `ultralytics`
* `kagglehub`
* `kaggle` (for `files.upload()` in Colab context to upload `kaggle.json`)
* `IPython.display` (for displaying video in Colab)
* `base64` (standard library)
* `collections` (standard library)
* `re` (standard library)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
