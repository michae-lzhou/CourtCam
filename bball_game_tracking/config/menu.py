import tkinter as tk
from tkinter import messagebox, ttk
import sys

class ConfigurationMenu:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Ball Tracking Configuration")
        self.root.geometry("500x600")  # Width x Height
        self.root.resizable(False, False)
        
        # Set default values
        self.processing_device = tk.StringVar(value="GPU")
        self.quality = tk.StringVar(value="Original")
        self.start_program = False
        
        self._create_widgets()
        self._center_window()

    def on_closing(self):
        """Handle window close button (X) event"""
        self.config = None  # Or set to default configuration if needed
        try:
            self.root.quit()  # Stop the mainloop
            self.root.destroy()  # Destroy the window
        except:
            pass  # Suppress any destruction-related errors
        
    def _center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def _create_widgets(self):
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Basketball Tracking Configuration",
            font=('Helvetica', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Processing Device Selection
        device_label = ttk.Label(
            main_frame,
            text="Processing Device:",
            font=('Helvetica', 10)
        )
        device_label.grid(row=1, column=0, sticky=tk.W, pady=(0, 5))
        
        device_combo = ttk.Combobox(
            main_frame,
            textvariable=self.processing_device,
            values=["GPU", "CPU"],
            state="readonly",
            width=30
        )
        device_combo.grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=(0, 15))
        
        # Quality Selection
        quality_label = ttk.Label(
            main_frame,
            text="Processing Quality:",
            font=('Helvetica', 10)
        )
        quality_label.grid(row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        quality_combo = ttk.Combobox(
            main_frame,
            textvariable=self.quality,
            values=["Original", "Fast"],
            state="readonly",
            width=30
        )
        quality_combo.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=(0, 20))
        
        # Information Text
        info_frame = ttk.Frame(main_frame)
        info_frame.grid(row=5, column=0, columnspan=2, pady=(0, 20))
        
        info_text = """
Basketball Game Tracker:

This deep-learning program lets parents focus on their child’s performance by simulating dynamic camera panning from a wide-angle recording at half-court.
Disclaimer: Performance may vary due to limited training data.

Processing Device:
• GPU (~7x speed) - Recommended for faster processing (requires CUDA)
• CPU (~0.85x speed) - Slower but universally compatible

Quality Settings:
• Original - Full quality processing; runtime scales with original FPS
• Fast (30 FPS) - Faster processing with fixed video quality.

Note: If GPU processing fails, the program will automatically
fall back to CPU processing.
        """
        
        info_label = ttk.Label(
            info_frame,
            text=info_text,
            justify=tk.LEFT,
            wraplength=450
        )
        info_label.pack(fill=tk.X, padx=10)
        
        # Buttons Frame
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=6, column=0, columnspan=2, pady=(20, 0))
        
        # Cancel Button
        cancel_btn = ttk.Button(
            button_frame,
            text="Cancel",
            command=self._on_cancel,
            width=15
        )
        cancel_btn.pack(side=tk.LEFT, padx=10)
        
        # Start Button
        start_btn = ttk.Button(
            button_frame,
            text="Start",
            command=self._on_start,
            width=15,
            style="Accent.TButton"
        )
        start_btn.pack(side=tk.LEFT, padx=10)
        
        # Create accent style for the Start button
        style = ttk.Style()
        style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))
        
    def _on_start(self):
        self.start_program = True
        self.root.quit()
        
    def _on_cancel(self):
        self.start_program = False
        self.root.quit()
        
    def get_configuration(self):
        """Run the menu and return the configuration"""
        try:
            self.root.mainloop()
        except:
            # Handle any mainloop exceptions
            self.start_program = False
            return None
            
        if not self.start_program:
            # User clicked X or cancelled
            try:
                self.root.destroy()
            except:
                pass  # Suppress destroy errors
            sys.exit(0)
            
        # Get configuration if program should start
        try:
            config = {
                'processing_device': self.processing_device.get(),
                'quality': self.quality.get()
            }
            self.root.destroy()
            return config
        except Exception as e:
            print(f"Error getting configuration: {e}")
            try:
                self.root.destroy()
            except:
                pass
            sys.exit(1)

def show_configuration_menu():
    """Shows the configuration menu and returns the user's choices"""
    menu = ConfigurationMenu()
    config = menu.get_configuration()
    return config
