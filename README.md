# face_recognition_with_yolo_and_facenet


![performance](https://github.com/mohanapavan/face_recognition_with_yolo_and_facenet/blob/main/output/output_face.png?raw=true)


## 🚀 Overview 

A robust face recognition system that:  
1. **Detects faces** using YOLO (with OpenCV fallback).  
2. **Generates embeddings** using FaceNet (or `face_recognition`/basic features as fallback).  
3. **Recognizes celebrities** by comparing embeddings against a pre-built database.


## 🧪 Test Results


#### Key Metrics

- 📈 Accuracy: 100.00%
- ⏱ Avg Confidence Score: 0.86
- ⚡ False Positives: 0


## Training Phase (Build Database)

![Training Phase](https://github.com/mohanapavan/face_recognition_with_yolo_and_facenet/blob/main/images/training%20phase.png?raw=true)


## Recognition  Phase (Build Database)

![Recognition  Phase](https://github.com/mohanapavan/face_recognition_with_yolo_and_facenet/blob/main/images/recog%20phase.png?raw=true)

## 🔄 Backup Systems & Key Behaviors

1. Cascade Model Loading

- The system attempts to load models in this order:
- 1. Primary: Ultralytics YOLO (latest version)
- 2. Fallback 1: PyTorch with `weights_only=False` (for custom YOLO)
- 3. Fallback 2: OpenCV Haar Cascade (if YOLO fails completely)

2. Face Detection
- YOLO Behavior:

- Only uses the top prediction (highest confidence face) per image

- Confidence threshold: Default 0.5 (adjustable in code)

- top_box = boxes_data[np.argsort(boxes_data[:, 4])[::-1][0]]


3. Database Construction
Mean Embedding Strategy:

- Takes first 5 images per celebrity

- Generates individual FaceNet embeddings (512-D)

- Stores the arithmetic mean of these 5 embeddings

- avg_embedding = np.mean([E1, E2, E3, E4, E5], axis=0)


4. Recognition Fallbacks

If FaceNet fails, tries:

1. face_recognition library (128-D embeddings)
2. Basic feature extraction (histograms + gradients)


## ⚠️ Current Limitations
Mean Embedding Sensitivity:

1. May underperform if:

- Training images have inconsistent lighting/angles

- Heavy occlusions (sunglasses/masks)


2. Top-Prediction Constraint:

- Cannot handle multi-face recognition in single image

3. Fallback Accuracy:

OpenCV/face_recognition are significantly less accurate than YOLO+FaceNet


