import os
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageDraw, ImageTk
from torchvision import transforms

class WaldoDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Waldo Detector")
        self.root.geometry("1200x800")

        # Paths
        self.bg_dir = "background"  # Dataset background directory
        self.object_dir = "objects"

        # State variables
        self.background_image = None
        self.waldo_image = None
        self.current_position = [0, 0]
        self.dragging = False
        self.model = None

        # Ensure directories exist
        os.makedirs(self.bg_dir, exist_ok=True)
        os.makedirs(self.object_dir, exist_ok=True)

        self.setup_ui()
        self.load_model()

    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Left panel for controls
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        # Background selection section
        ttk.Label(left_panel, text="Select Background", font=('Arial', 12, 'bold')).pack(pady=(0, 10))
        ttk.Button(left_panel, text="Browse Backgrounds", command=self.browse_backgrounds).pack(fill=tk.X)

        # Object selection section
        ttk.Label(left_panel, text="Select Object", font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        ttk.Button(left_panel, text="Browse Objects", command=self.browse_objects).pack(fill=tk.X)

        # Waldo placement section
        ttk.Label(left_panel, text="Place Object", font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        ttk.Label(left_panel, text="Drag and drop object on the background").pack()

        # Detection confidence
        ttk.Label(left_panel, text="Detection Confidence", font=('Arial', 12, 'bold')).pack(pady=(20, 10))
        self.confidence_var = tk.DoubleVar(value=0.5)
        confidence_slider = ttk.Scale(
            left_panel,
            from_=0.1, to=1.0,
            variable=self.confidence_var,
            orient=tk.HORIZONTAL
        )
        confidence_slider.pack(fill=tk.X)

        # Detect button
        ttk.Button(left_panel, text="Detect Object", command=self.detect_waldo).pack(fill=tk.X, pady=20)

        # Results area
        self.results_text = tk.Text(left_panel, height=10, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X)

        # Canvas for image display
        self.canvas = tk.Canvas(main_frame, bg='lightgray')
        self.canvas.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10)

        # Canvas event bindings
        self.canvas.bind("<ButtonPress-1>", self.start_drag)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drag)

    def browse_objects(self):
        """Browse and select object from dataset"""
        objects = [f for f in os.listdir(self.object_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not objects:
            messagebox.showinfo("No Objects", "No objects found in the dataset.")
            return

        # Create selection window
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Object")
        select_window.geometry("800x600")

        canvas = tk.Canvas(select_window)
        scrollbar = ttk.Scrollbar(select_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display object thumbnails
        for i, obj_file in enumerate(objects):
            obj_path = os.path.join(self.object_dir, obj_file)
            img = Image.open(obj_path)
            img.thumbnail((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            btn = ttk.Button(
                scrollable_frame,
                image=photo,
                command=lambda path=obj_path: self.select_object(path, select_window)
            )
            btn.image = photo  # Keep reference
            btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)

    def select_object(self, path, window):
        """Load selected object"""
        self.waldo_image = Image.open(path)
        self.waldo_image = self.waldo_image.resize((100, 150), Image.LANCZOS)
        window.destroy()

        # If background is already selected, update canvas
        if self.background_image:
            self.update_canvas()

        # Update results text
        object_name = os.path.basename(path)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Object Selected: {object_name}")

    def load_model(self):
        """Load the Waldo detection model"""
        try:
            # Placeholder for model loading logic
            print("Loading Waldo detection model...")
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, "Model loaded successfully")
        except Exception as e:
            messagebox.showerror("Model Error", str(e))

    def browse_backgrounds(self):
        """Browse and select background from dataset"""
        backgrounds = [f for f in os.listdir(self.bg_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        if not backgrounds:
            messagebox.showinfo("No Backgrounds", "No backgrounds found in the dataset.")
            return

        # Create selection window
        select_window = tk.Toplevel(self.root)
        select_window.title("Select Background")
        select_window.geometry("800x600")

        canvas = tk.Canvas(select_window)
        scrollbar = ttk.Scrollbar(select_window, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Display background thumbnails
        for i, bg_file in enumerate(backgrounds):
            bg_path = os.path.join(self.bg_dir, bg_file)
            img = Image.open(bg_path)
            img.thumbnail((200, 200), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            btn = ttk.Button(
                scrollable_frame,
                image=photo,
                command=lambda path=bg_path: self.select_background(path, select_window)
            )
            btn.image = photo  # Keep reference
            btn.grid(row=i // 3, column=i % 3, padx=10, pady=10)

    def select_background(self, path, window):
        """Load selected background"""
        self.background_image = Image.open(path)
        self.update_canvas()
        window.destroy()

    def update_canvas(self):
        """Update canvas with current scene"""
        if self.background_image:
            # Resize canvas
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Resize background to fit canvas while maintaining aspect ratio
            bg_img = self.background_image.copy()
            bg_img.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

            # Convert to PhotoImage for display
            self.canvas_photo = ImageTk.PhotoImage(bg_img)

            # Clear and redraw canvas
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_photo)

    def start_drag(self, event):
        """Start dragging Waldo"""
        self.dragging = True
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def drag(self, event):
        """Handle dragging of Waldo"""
        if self.dragging and self.waldo_image:
            # Update current position
            self.current_position = [event.x - 50, event.y - 75]
            self.place_waldo()

    def stop_drag(self, event):
        """Stop dragging"""
        self.dragging = False

    def place_waldo(self):
        """Place Waldo on the background"""
        if not self.background_image or not self.waldo_image:
            return

        # Create copy of background
        scene = self.background_image.copy()

        # Paste Waldo
        scene.paste(
            self.waldo_image,
            (self.current_position[0], self.current_position[1]),
            self.waldo_image
        )

        # Display updated scene
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        scene.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

        self.canvas_photo = ImageTk.PhotoImage(scene)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_photo)

    def detect_waldo(self):
        """Detect Waldo with confidence threshold"""
        if not self.background_image or not self.waldo_image:
            messagebox.showinfo("Missing Data", "Please select a background and place Waldo")
            return

        try:
            # Simulate detection
            confidence = 0.75  # Placeholder
            x, y = self.current_position

            # Draw detection result
            result_scene = self.background_image.copy()
            draw = ImageDraw.Draw(result_scene)

            # Draw bounding box
            draw.rectangle(
                [x, y, x + 100, y + 150],
                outline="red",
                width=3
            )

            # Display updated scene
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            result_scene.thumbnail((canvas_width, canvas_height), Image.LANCZOS)

            self.canvas_photo = ImageTk.PhotoImage(result_scene)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.canvas_photo)

            # Update results text
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"Detection Confidence: {confidence:.2f}")
        except Exception as e:
            messagebox.showerror("Detection Error", str(e))


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = WaldoDetectorApp(root)
    root.mainloop()