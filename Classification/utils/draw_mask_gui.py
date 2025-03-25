import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
import os
from PIL import Image, ImageTk, ImageOps
import torchvision.transforms as transforms

# Constants (same as in your main script)
TARGET_SIZE = 256
CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = 8

# Custom Preprocessor (adapted for display, train=False)
class XRayPreprocessor:
    def __init__(self, train=False, target_size=256):
        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(CLAHE_GRID_SIZE, CLAHE_GRID_SIZE))
        self.target_size = target_size
        self.tensor_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        ])
        self.augment = None  # No augmentations for display

    def enhance_contrast(self, image):
        image_np = np.array(image)
        enhanced_image = self.clahe.apply(image_np)
        return Image.fromarray(enhanced_image)

    def resize_with_padding(self, image):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height
        
        if original_width > original_height:
            new_height = int(self.target_size / aspect_ratio)
            resized_image = image.resize((self.target_size, new_height), Image.Resampling.BICUBIC)
        else:
            new_width = int(self.target_size * aspect_ratio)
            resized_image = image.resize((new_width, self.target_size), Image.Resampling.BICUBIC)
        
        padded_image = ImageOps.pad(resized_image, (self.target_size, self.target_size), color=0, centering=(0.5, 0.5))
        return padded_image.convert('L')

    def __call__(self, image):
        enhanced_image = self.enhance_contrast(image)
        padded_image = self.resize_with_padding(enhanced_image)
        return padded_image  # Return PIL Image for display

class MaskDrawerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Draw ROI Mask for X-Ray Image")

        # Variables for drawing
        self.image_path = None
        self.image = None  # Original image after preprocessing
        self.mask = None
        self.display_image = None
        self.start_x, self.start_y = None, None
        self.rect_drawn = False
        self.rect_coords = None
        self.preprocessor = XRayPreprocessor(train=False, target_size=TARGET_SIZE)

        # Create GUI elements
        self.load_button = tk.Button(root, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=5)

        self.canvas = tk.Canvas(root, width=TARGET_SIZE, height=TARGET_SIZE)
        self.canvas.pack(pady=5)

        self.save_button = tk.Button(root, text="Save Mask", command=self.save_mask, state=tk.DISABLED)
        self.save_button.pack(pady=5)

        # Bind mouse events to the canvas
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_rectangle)
        self.canvas.bind("<ButtonRelease-1>", self.end_drawing)

    def load_image(self):
        # Open file dialog to select an image
        self.image_path = filedialog.askopenfilename(
            title="Select X-Ray Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not self.image_path:
            return

        # Load and preprocess the image using XRayPreprocessor
        raw_image = Image.open(self.image_path).convert('L')
        self.image = self.preprocessor(raw_image)  # Apply preprocessing (no augmentations)
        self.image_np = np.array(self.image)  # Convert to numpy for mask and drawing
        self.mask = np.zeros_like(self.image_np)  # Initialize mask
        self.display_image = self.image_np.copy()

        # Convert image to PhotoImage for Tkinter
        self.display_image_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_GRAY2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.display_image_rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.save_button.config(state=tk.NORMAL)

    def start_drawing(self, event):
        self.start_x, self.start_y = event.x, event.y
        self.rect_drawn = False

    def draw_rectangle(self, event):
        # Clear previous rectangle
        self.display_image = self.image_np.copy()
        # Draw new rectangle
        cv2.rectangle(self.display_image, (self.start_x, self.start_y), (event.x, event.y), 255, 2)
        # Update canvas
        self.display_image_rgb = cv2.cvtColor(self.display_image, cv2.COLOR_GRAY2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(self.display_image_rgb))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def end_drawing(self, event):
        # Record the final rectangle coordinates
        end_x, end_y = event.x, event.y
        self.rect_coords = (self.start_x, self.start_y, end_x, end_y)
        # Update mask
        self.mask = np.zeros_like(self.image_np)
        cv2.rectangle(self.mask, (self.start_x, self.start_y), (end_x, end_y), 255, -1)
        self.rect_drawn = True

    def save_mask(self):
        if not self.rect_drawn or self.image_path is None:
            tk.messagebox.showwarning("Warning", "Please draw a rectangle on the image first!")
            return

        # Create output directory if it doesn't exist
        output_dir = "dataset/masks"
        os.makedirs(output_dir, exist_ok=True)

        # Generate mask filename based on the image filename
        image_filename = os.path.basename(self.image_path)
        mask_filename = image_filename.replace(os.path.splitext(image_filename)[1], "_mask.png")
        mask_path = os.path.join(output_dir, mask_filename)

        # Save the mask
        cv2.imwrite(mask_path, self.mask)
        tk.messagebox.showinfo("Success", f"Mask saved to {mask_path}")

        # Reset for the next image
        self.image_path = None
        self.image = None
        self.image_np = None
        self.mask = None
        self.display_image = None
        self.rect_drawn = False
        self.canvas.delete("all")
        self.save_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDrawerApp(root)
    root.mainloop()
