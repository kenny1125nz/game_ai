import win32gui
import win32api
import win32con
import threading
import time

RUNE_COLOR = win32api.RGB(255, 164, 0)

class Overlay:
    def __init__(self, offset=(0, 0)):
        self.detections = []
        self._offset = offset
        self._lock = threading.Lock()
        self.ox, self.oy = offset
        
        threading.Thread(target=self._overlay_thread, daemon=True).start()
    
    def _overlay_thread(self):
        while True:
            with self._lock:
                current_batch = list(self.detections)

            # --- THE "CLEAN" FIX ---
            # If there are no detections, we MUST tell Windows to wipe the screen
            # otherwise the last drawn boxes will stay there forever.
            if not current_batch:
                # InvalidateRect(0, None, True) refreshes the ENTIRE desktop
                win32gui.InvalidateRect(0, None, True)
                time.sleep(0.1) 
                continue

            # Open DC for drawing
            hdc = win32gui.GetWindowDC(0)
            
            # Setup Pen and Text
            pen = win32gui.CreatePen(win32con.PS_SOLID, 2, RUNE_COLOR)
            old_pen = win32gui.SelectObject(hdc, pen)
            win32gui.SetTextColor(hdc, RUNE_COLOR)
            win32gui.SetBkMode(hdc, win32con.TRANSPARENT)
            
            # Use NULL_BRUSH to keep the box transparent
            old_brush = win32gui.SelectObject(hdc, win32gui.GetStockObject(win32con.NULL_BRUSH))

            for d in current_batch:
                try:
                    text = d["text"]
                    x, y, w, h = d["roi"]
                    x1, y1 = x + self.ox, y + self.oy
                    x2, y2 = x1 + w, y1 + h

                    # Draw Box
                    win32gui.Rectangle(hdc, x1, y1, x2, y2)

                    # Draw Text
                    t_h = 20
                    t_rect = (x1, y2, x1 + 500, y2 + t_h) if y1 < t_h else (x1, y1 - t_h, x1 + 500, y1)
                    win32gui.DrawText(hdc, text, -1, t_rect, win32con.DT_LEFT | win32con.DT_NOCLIP)
                except: continue

            # Cleanup GDI handles
            win32gui.SelectObject(hdc, old_pen)
            win32gui.SelectObject(hdc, old_brush)
            win32gui.DeleteObject(pen)
            win32gui.ReleaseDC(0, hdc)

            # This forces a refresh of the area we just drew after a short delay
            # to help prevent "stuck" boxes if the detection changes.
            time.sleep(0.01)

    def update(self, detections):
        with self._lock:
            # If the new list is empty and the old one wasn't, 
            # we trigger a one-time full refresh to clear the "ghosts"
            if not detections and self.detections:
                win32gui.InvalidateRect(0, None, True)
            self.detections = detections