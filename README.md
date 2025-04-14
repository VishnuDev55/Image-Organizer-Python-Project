# Image-Organizer-Python-Project
## Overview
The **Face Recognition and Image Searcher** is a Python-based desktop application designed to index, cluster, and search for faces across a collection of images. It uses face recognition technology to identify and group similar faces, allowing users to browse unique faces and view all images containing a selected face. The system is built with DeepFace for face detection and embedding extraction, SQLite for storing face data, and Tkinter for a user-friendly graphical interface.

This project is ideal for organizing large photo collections, finding specific individuals across images, or exploring face similarity in datasets.

## Features
- **Face Indexing**: Automatically detects and extracts face embeddings from images in a specified folder.
- **Clustering**: Groups similar faces into clusters based on embedding similarity, with an adjustable threshold.
- **GUI Search**:
  - Displays unique faces (cropped) on the left panel, each representing a cluster.
  - Shows all matching full images (with highlighted faces) on the right panel when a face is clicked.
- **Threshold Adjustment**: Includes a slider to tweak the clustering similarity threshold (0.4 to 0.8) and recluster faces dynamically.
- **Database Storage**: Stores face metadata (embeddings, bounding boxes, file paths) in a SQLite database for efficient querying.
- **Error Handling**: Robustly handles invalid images, missing files, and detection failures.

## How It Works
1. **Indexing**:
   - The `index_faces.py` script scans an `images` folder for `.jpg`, `.jpeg`, or `.png` files.
   - DeepFace detects faces, extracts 128D embeddings, and saves bounding box data.
   - Cropped face images are stored in a `faces` folder, and metadata is saved to `db/faces.db`.

2. **Clustering**:
   - Faces are grouped by comparing embeddings using Euclidean distance.
   - Each cluster is assigned a representative face, and cluster IDs are stored in the database.

3. **GUI**:
   - The left panel shows cropped faces (one per cluster), labeled with a face ID.
   - Clicking a face queries the database for all images in the same cluster, displaying them on the right with red rectangles around the matched faces.
   - A slider allows adjusting the clustering threshold, updating the displayed faces in real-time.

## Requirements
- **Python**: 3.8 or higher
- **Dependencies**:
  ```bash
  pip install deepface numpy sqlite3 pillow tkinter deepface 
