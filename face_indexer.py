import os
import sys
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import face_recognition
from deepface import DeepFace

class FaceIndexer:
    def __init__(self, base_dir=None):
        # Set up directories
        if base_dir is None:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            self.base_dir = base_dir
            
        self.faces_dir = os.path.join(self.base_dir, "faces")
        self.db_path = os.path.join(self.base_dir, "face_database.db")
        
        # Create directories if they don't exist
        os.makedirs(self.faces_dir, exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # For callback functions
        self.progress_callback = None
        self.status_callback = None
    
    def set_callbacks(self, progress_callback=None, status_callback=None):
        """Set callback functions for progress updates"""
        self.progress_callback = progress_callback
        self.status_callback = status_callback
    
    def init_database(self):
        """Initialize the SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY,
            face_path TEXT NOT NULL,
            quality REAL DEFAULT 0,
            date_added TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY,
            image_path TEXT NOT NULL,
            date_added TEXT
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS face_image_map (
            face_id INTEGER,
            image_id INTEGER,
            face_location TEXT,
            PRIMARY KEY (face_id, image_id),
            FOREIGN KEY (face_id) REFERENCES faces(id),
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def process_images(self, file_paths):
        """Process a list of image paths to detect and extract faces"""
        total_files = len(file_paths)
        
        for i, file_path in enumerate(file_paths):
            # Update progress
            if self.progress_callback:
                progress_value = int((i / total_files) * 100)
                self.progress_callback(progress_value)
            
            # Update status
            if self.status_callback:
                self.status_callback(f"Processing image {i+1}/{total_files}: {os.path.basename(file_path)}")
            
            # Process the image
            self.process_single_image(file_path)
        
        # Complete
        if self.progress_callback:
            self.progress_callback(100)
        
        if self.status_callback:
            self.status_callback("Processing complete!")
    
    def process_single_image(self, image_path):
        """Process a single image to detect and extract faces"""
        try:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                if self.status_callback:
                    self.status_callback(f"Could not read image: {image_path}")
                return
            
            # Convert BGR to RGB (for face_recognition library)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                if self.status_callback:
                    self.status_callback(f"No faces found in: {os.path.basename(image_path)}")
                return
            
            # Store the image in the database
            image_id = self.store_image_in_db(image_path)
            
            # Process each detected face
            for face_location in face_locations:
                top, right, bottom, left = face_location
                
                # Expand the boundaries slightly to get more of the face
                height, width = image.shape[:2]
                top = max(0, top - int((bottom - top) * 0.2))
                bottom = min(height, bottom + int((bottom - top) * 0.1))
                left = max(0, left - int((right - left) * 0.1))
                right = min(width, right + int((right - left) * 0.1))
                
                # Crop the face
                face_img = image[top:bottom, left:right]
                
                # Calculate face quality (brightness and contrast)
                quality = self.calculate_face_quality(face_img)
                
                # Get face encoding for comparison
                face_encoding = face_recognition.face_encodings(rgb_image, [face_location])[0]
                
                # Check if this face matches any existing faces
                matched_face_id = self.find_matching_face(face_encoding)
                
                if matched_face_id is None:
                    # This is a new unique face
                    face_filename = f"face_{datetime.now().strftime('%Y%m%d%H%M%S%f')}.jpg"
                    face_path = os.path.join(self.faces_dir, face_filename)
                    cv2.imwrite(face_path, face_img)
                    
                    # Store the new face in the database
                    face_id = self.store_face_in_db(face_path, quality)
                    
                    # Map face to image
                    self.map_face_to_image(face_id, image_id, face_location)
                    
                    if self.status_callback:
                        self.status_callback(f"New face found and saved: {face_filename}")
                else:
                    # This face matches an existing face
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    
                    # Get quality of the existing face
                    cursor.execute("SELECT quality, face_path FROM faces WHERE id = ?", (matched_face_id,))
                    existing_quality, existing_face_path = cursor.fetchone()
                    
                    # Map face to image
                    self.map_face_to_image(matched_face_id, image_id, face_location)
                    
                    # If this face is better quality, replace the existing one
                    if quality > existing_quality:
                        # Save the new face
                        cv2.imwrite(existing_face_path, face_img)
                        
                        # Update the quality in the database
                        cursor.execute("UPDATE faces SET quality = ? WHERE id = ?", (quality, matched_face_id))
                        conn.commit()
                        
                        if self.status_callback:
                            self.status_callback(f"Updated face with better quality: {os.path.basename(existing_face_path)}")
                    
                    conn.close()
        
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            if self.status_callback:
                self.status_callback(f"Error: {str(e)}")
    
    def calculate_face_quality(self, face_img):
        """Calculate face quality based on brightness and contrast"""
        # Convert to grayscale for calculations
        if face_img.size == 0:
            return 0
            
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(gray)
        
        # Calculate contrast (standard deviation)
        contrast = np.std(gray)
        
        # Calculate face size
        size = face_img.shape[0] * face_img.shape[1]
        
        # Combine metrics into a quality score (adjust weights as needed)
        # Ideal brightness is around 120-150
        brightness_score = 1 - abs(brightness - 135) / 135
        
        # Higher contrast is generally better, up to a point
        contrast_score = min(contrast / 80, 1.0)
        
        # Larger face size is better
        size_score = min(size / 40000, 1.0)
        
        # Weighted quality score
        quality = (brightness_score * 0.4) + (contrast_score * 0.4) + (size_score * 0.2)
        
        return quality
    
    def find_matching_face(self, face_encoding, tolerance=0.6):
        """Find if this face matches any existing face in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, face_path FROM faces")
        existing_faces = cursor.fetchall()
        conn.close()
        
        for face_id, face_path in existing_faces:
            try:
                # Load the existing face
                existing_face = cv2.imread(face_path)
                existing_rgb = cv2.cvtColor(existing_face, cv2.COLOR_BGR2RGB)
                
                # Get encoding for the existing face
                existing_encoding = face_recognition.face_encodings(existing_rgb)
                
                if len(existing_encoding) > 0:
                    # Compare faces
                    distance = face_recognition.face_distance([existing_encoding[0]], face_encoding)[0]
                    
                    if distance < tolerance:
                        return face_id
            except Exception as e:
                print(f"Error comparing face {face_id}: {str(e)}")
        
        return None
    
    def store_image_in_db(self, image_path):
        """Store image path in the database and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if image already exists
        cursor.execute("SELECT id FROM images WHERE image_path = ?", (image_path,))
        result = cursor.fetchone()
        
        if result:
            conn.close()
            return result[0]
        
        # Insert new image
        cursor.execute(
            "INSERT INTO images (image_path, date_added) VALUES (?, ?)",
            (image_path, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        image_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return image_id
    
    def store_face_in_db(self, face_path, quality):
        """Store face path in the database and return its ID"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT INTO faces (face_path, quality, date_added) VALUES (?, ?, ?)",
            (face_path, quality, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        
        face_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return face_id
    
    def map_face_to_image(self, face_id, image_id, face_location):
        """Map face to image in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store face location as string
        location_str = ",".join(map(str, face_location))
        
        # Check if mapping already exists
        cursor.execute(
            "SELECT 1 FROM face_image_map WHERE face_id = ? AND image_id = ?",
            (face_id, image_id)
        )
        
        if not cursor.fetchone():
            # Insert new mapping
            cursor.execute(
                "INSERT INTO face_image_map (face_id, image_id, face_location) VALUES (?, ?, ?)",
                (face_id, image_id, location_str)
            )
            
            conn.commit()
            
        conn.close()
    
    def get_all_faces(self):
        """Return all faces from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, face_path FROM faces ORDER BY id")
        faces = cursor.fetchall()
        
        conn.close()
        return faces
    
    def get_images_with_face(self, face_id):
        """Get all images containing a specific face"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT i.image_path, m.face_location 
            FROM images i
            JOIN face_image_map m ON i.id = m.image_id
            WHERE m.face_id = ?
        """, (face_id,))
        
        images = cursor.fetchall()
        conn.close()
        
        return images

# Example usage
if __name__ == "__main__":
    # Basic command-line interface for testing
    indexer = FaceIndexer()
    
    if len(sys.argv) > 1:
        # Process files from command line arguments
        indexer.process_images(sys.argv[1:])
    else:
        print("Usage: python face_indexer.py [image_files...]")