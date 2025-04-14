import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import threading
import queue
from face_indexer import FaceIndexer

class FaceGalleryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Gallery")
        self.root.geometry("1200x700")
        self.root.configure(bg="#f0f0f0")
        
        # Initialize the face indexer
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.indexer = FaceIndexer(self.base_dir)
        
        # Status variables
        self.processing = False
        self.status_queue = queue.Queue()
        
        # Create GUI components
        self.create_widgets()
        
        # Set callbacks for the indexer
        self.indexer.set_callbacks(
            progress_callback=self.update_progress,
            status_callback=lambda msg: self.status_queue.put(msg)
        )
        
        # Start a thread to update the status message
        self.update_status()
    
    def create_widgets(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = tk.Frame(main_frame, bg="#e0e0e0", relief=tk.RIDGE, bd=2)
        control_frame.pack(fill=tk.X, pady=5)
        
        # Buttons for adding images
        self.add_btn = tk.Button(control_frame, text="Add Images", command=self.add_images,
                           bg="#4CAF50", fg="white", font=("Arial", 10, "bold"), padx=10)
        self.add_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.add_folder_btn = tk.Button(control_frame, text="Add Folder", command=self.add_folder,
                             bg="#2196F3", fg="white", font=("Arial", 10, "bold"), padx=10)
        self.add_folder_btn.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Status label
        self.status_label = tk.Label(control_frame, text="Ready", bg="#e0e0e0", font=("Arial", 9))
        self.status_label.pack(side=tk.RIGHT, padx=10, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)
        
        # Panel for faces and images
        panel_frame = tk.Frame(main_frame, bg="#f0f0f0")
        panel_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - unique faces
        faces_frame = tk.LabelFrame(panel_frame, text="Unique Faces", bg="#f0f0f0", font=("Arial", 10, "bold"))
        faces_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.faces_canvas = tk.Canvas(faces_frame, bg="white")
        scrollbar = tk.Scrollbar(faces_frame, orient="vertical", command=self.faces_canvas.yview)
        self.faces_frame_inner = tk.Frame(self.faces_canvas, bg="white")
        
        self.faces_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.faces_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.faces_canvas.create_window((0, 0), window=self.faces_frame_inner, anchor='nw')
        self.faces_frame_inner.bind("<Configure>", lambda e: self.faces_canvas.configure(scrollregion=self.faces_canvas.bbox("all")))
        
        # Right panel - images with selected face
        images_frame = tk.LabelFrame(panel_frame, text="Images with Selected Face", bg="#f0f0f0", font=("Arial", 10, "bold"))
        images_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.images_canvas = tk.Canvas(images_frame, bg="white")
        scrollbar_images = tk.Scrollbar(images_frame, orient="vertical", command=self.images_canvas.yview)
        self.images_frame_inner = tk.Frame(self.images_canvas, bg="white")
        
        self.images_canvas.configure(yscrollcommand=scrollbar_images.set)
        scrollbar_images.pack(side=tk.RIGHT, fill=tk.Y)
        self.images_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.images_canvas.create_window((0, 0), window=self.images_frame_inner, anchor='nw')
        self.images_frame_inner.bind("<Configure>", lambda e: self.images_canvas.configure(scrollregion=self.images_canvas.bbox("all")))
        
        # Load existing faces
        self.load_faces()
    
    def update_status(self):
        """Update the status message from the queue"""
        try:
            while not self.status_queue.empty():
                message = self.status_queue.get(0)
                self.status_label.config(text=message)
                self.root.update_idletasks()
        finally:
            self.root.after(100, self.update_status)
    
    def update_progress(self, value):
        """Update the progress bar"""
        self.progress['value'] = value
        self.root.update_idletasks()
    
    def add_images(self):
        """Add individual images to process"""
        if self.processing:
            messagebox.showinfo("Processing", "Please wait until current processing is complete.")
            return
        
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*"))
        )
        
        if file_paths:
            self.process_images(file_paths)
    
    def add_folder(self):
        """Add all images from a folder to process"""
        if self.processing:
            messagebox.showinfo("Processing", "Please wait until current processing is complete.")
            return
        
        folder_path = filedialog.askdirectory(title="Select Folder with Images")
        
        if folder_path:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            file_paths = []
            
            for root, _, files in os.walk(folder_path):
                for file in files:
                    if any(file.lower().endswith(ext) for ext in image_extensions):
                        file_paths.append(os.path.join(root, file))
            
            if file_paths:
                self.process_images(file_paths)
            else:
                messagebox.showinfo("No Images", "No image files found in the selected folder.")
    
    def process_images(self, file_paths):
        """Process images in a separate thread"""
        self.processing = True
        self.progress['value'] = 0
        
        # Start processing in a new thread
        threading.Thread(
            target=self._process_images_thread,
            args=(file_paths,),
            daemon=True
        ).start()
    
    def _process_images_thread(self, file_paths):
        """Background thread for processing images"""
        try:
            # Process all images
            self.indexer.process_images(file_paths)
            
            # Update the faces display when done
            self.root.after(0, self.load_faces)
        
        except Exception as e:
            self.status_queue.put(f"Error: {str(e)}")
        
        finally:
            self.processing = False
    
    def load_faces(self):
        """Load and display all unique faces"""
        # Clear the faces frame
        for widget in self.faces_frame_inner.winfo_children():
            widget.destroy()
        
        # Get faces from database
        faces = self.indexer.get_all_faces()
        
        # Display faces
        if not faces:
            label = tk.Label(self.faces_frame_inner, text="No faces found", bg="white")
            label.pack(pady=20)
            return
        
        # Create a frame for faces grid
        faces_grid = tk.Frame(self.faces_frame_inner, bg="white")
        faces_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Calculate dimensions for thumbnails
        thumbnail_size = 100
        columns = max(1, self.faces_canvas.winfo_width() // (thumbnail_size + 10))
        if columns < 1:
            columns = 3  # Default if width is not yet available
        
        # Display faces in a grid
        for i, (face_id, face_path) in enumerate(faces):
            row = i // columns
            col = i % columns
            
            try:
                # Create face frame
                face_frame = tk.Frame(faces_grid, bg="white", relief=tk.RIDGE, bd=1)
                face_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Load and resize image
                img = Image.open(face_path)
                img = img.resize((thumbnail_size, thumbnail_size), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Store reference to prevent garbage collection
                face_frame.image = photo
                
                # Display face
                lbl = tk.Label(face_frame, image=photo, bg="white")
                lbl.pack(fill=tk.BOTH)
                
                # Add a face ID label
                id_lbl = tk.Label(face_frame, text=f"Face #{face_id}", bg="white", font=("Arial", 8))
                id_lbl.pack()
                
                # Add click event to show related images
                lbl.bind("<Button-1>", lambda e, fid=face_id: self.show_images_with_face(fid))
                
            except Exception as e:
                print(f"Error loading face {face_id}: {str(e)}")
    
    def show_images_with_face(self, face_id):
        """Show all images containing the selected face"""
        # Clear the images frame
        for widget in self.images_frame_inner.winfo_children():
            widget.destroy()
        
        # Get images with this face
        images = self.indexer.get_images_with_face(face_id)
        
        if not images:
            label = tk.Label(self.images_frame_inner, text="No images found with this face", bg="white")
            label.pack(pady=20)
            return
        
        # Create a frame for images grid
        images_grid = tk.Frame(self.images_frame_inner, bg="white")
        images_grid.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Calculate dimensions for thumbnails
        thumbnail_size = 150
        columns = max(1, self.images_canvas.winfo_width() // (thumbnail_size + 10))
        if columns < 1:
            columns = 2  # Default if width is not yet available
        
        # Display images in a grid
        for i, (image_path, face_location) in enumerate(images):
            row = i // columns
            col = i % columns
            
            try:
                # Create image frame
                img_frame = tk.Frame(images_grid, bg="white", relief=tk.RIDGE, bd=1)
                img_frame.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                
                # Load and resize image
                img = Image.open(image_path)
                img.thumbnail((thumbnail_size, thumbnail_size), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                
                # Store reference to prevent garbage collection
                img_frame.image = photo
                
                # Display image
                lbl = tk.Label(img_frame, image=photo, bg="white")
                lbl.pack(fill=tk.BOTH)
                
                # Add filename label
                filename = os.path.basename(image_path)
                if len(filename) > 20:
                    filename = filename[:18] + "..."
                name_lbl = tk.Label(img_frame, text=filename, bg="white", font=("Arial", 8))
                name_lbl.pack()
                
                # Double-click to open full image
                lbl.bind("<Double-Button-1>", lambda e, path=image_path: self.open_full_image(path))
                
            except Exception as e:
                print(f"Error loading image {image_path}: {str(e)}")
    
    def open_full_image(self, image_path):
        """Open the full image in the default image viewer"""
        try:
            # For Windows
            if sys.platform == "win32":
                os.startfile(image_path)
            # For macOS
            elif sys.platform == "darwin":
                os.system(f"open \"{image_path}\"")
            # For Linux
            else:
                os.system(f"xdg-open \"{image_path}\"")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open image: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceGalleryApp(root)
    root.mainloop()