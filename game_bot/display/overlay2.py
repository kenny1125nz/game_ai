import tkinter as tk
import threading
import time

COLOR="#00A4FF"

class Overlay:
    def __init__(self, offset=(0, 0)):
        self.ox, self.oy = offset
        self.detections = []
        self._lock = threading.Lock()
        self.initialized = False
        
        # Start the GUI in a background thread
        threading.Thread(target=self._run_gui, daemon=True).start()
        
        # Wait for the window to actually exist before returning
        while not self.initialized:
            time.sleep(0.1)

    def _run_gui(self):
        self.root = tk.Tk()
        
        # 1. Visual Properties: Borderless, TopMost, Click-Through
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        self.root.attributes("-disabled", True)  # Makes it click-through
        self.root.attributes("-transparentcolor", "black") # Black pixels = Invisible
        
        # Cover the whole screen
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_w}x{screen_h}+0+0")
        
        # 2. Setup Canvas
        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        
        self.initialized = True
        
        # 3. The "Sync" Loop: Checks if detections changed and redraws
        self._sync_with_detections()
        self.root.mainloop()

    def _sync_with_detections(self):
        """Internal loop that updates the canvas based on self.detections."""
        self.canvas.delete("all") # Clear everything
        
        with self._lock:
            current_batch = list(self.detections)

        for d in current_batch:
            try:
                text = d["text"]
                x, y, w, h = d["roi"]
                x1, y1 = x + self.ox, y + self.oy
                x2, y2 = x1 + w, y1 + h
                
                # Draw Box (Gold/Orange #FFA400)
                self.canvas.create_rectangle(x1, y1, x2, y2, outline=COLOR, width=1)
                
                # Logic for text placement (Above vs Below)
                ty = y1 - 2 if y1 > 20 else y2 + 15
                self.canvas.create_text(x1, ty, text=text, fill=COLOR, 
                                       anchor="sw", font=("Arial", 10))
            except (KeyError, TypeError):
                continue
        
        # Schedule the next sync (adjust ms for responsiveness vs CPU)
        self.root.after(30, self._sync_with_detections)

    def update(self, detections):
        """External Interface: Same as before."""
        with self._lock:
            self.detections = detections

# --- EXAMPLE USAGE ---
# overlay = Overlay(offset=(10, 10))
# overlay.update([{"text": "Ber Rune", "roi": (500, 500, 100, 50)}])