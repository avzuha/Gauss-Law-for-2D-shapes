import tkinter as tk
from tkinter import messagebox
import numpy as np
from matplotlib.path import Path

def vector_field(x, y):
    return np.array([x, y])

def flux_through_boundary(points):
    flux = 0.0
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i+1]
        mid = (p1 + p2) / 2.0
        F = vector_field(mid[0], mid[1])
        dl = p2 - p1
        length = np.linalg.norm(dl)
        if length > 0:
            normal = np.array([-dl[1], dl[0]]) / length
            flux += np.dot(F, normal) * length
    return abs(flux)

def divergence_theorem(points, base_resolution=200):
    path = Path(points)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    area = (x_max - x_min) * (y_max - y_min)
    resolution = int(base_resolution + np.sqrt(area) / 2)
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    dx = (x_max - x_min) / resolution
    dy = (y_max - y_min) / resolution

    div_integral = 0.0
    for x in xs:
        for y in ys:
            if path.contains_point((x, y)):
                div_integral += 2 * dx * dy
    return abs(div_integral)

def verify_gauss(points):
    if len(points) < 3:
        messagebox.showinfo("Result", "Not enough points to form a closed shape.")
        return
    if np.linalg.norm(points[0] - points[-1]) > 10:
        messagebox.showinfo("Result", "Shape is not closed. Please draw a closed boundary.")
        return
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0]])

    flux = flux_through_boundary(points)
    div_integral = divergence_theorem(points)

    diff = abs(flux - div_integral)
    rel_error = diff / max(div_integral, 1e-6)

    if rel_error > 0.05:
        div_integral = divergence_theorem(points, base_resolution=400)
        diff = abs(flux - div_integral)
        rel_error = diff / max(div_integral, 1e-6)

    if rel_error < 0.05:
        result_text = (f"Gauss's Law Verified!\n"
                       f"Flux ≈ {flux:.3f}\n"
                       f"Divergence integral ≈ {div_integral:.3f}\n"
                       f"Relative error = {rel_error:.3%}")
    else:
        result_text = (f"Large difference detected.\n"
                       f"Flux ≈ {flux:.3f}\n"
                       f"Divergence integral ≈ {div_integral:.3f}\n"
                       f"Relative error = {rel_error:.3%}\n"
                       f"(Likely due to irregular drawing)")
    messagebox.showinfo("Result", result_text)

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gauss's Law Drawing App")
        self.canvas = tk.Canvas(root, width=500, height=500, bg="white")
        self.canvas.pack()
        self.points = []
        self.canvas.bind("<B1-Motion>", self.paint)
        btn_frame = tk.Frame(root)
        btn_frame.pack()
        tk.Button(btn_frame, text="Rectangle", command=self.draw_rectangle).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Circle", command=self.draw_circle).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Oval", command=self.draw_oval).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Star", command=self.draw_star).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Pentagon", command=self.draw_pentagon).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Hexagon", command=self.draw_hexagon).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Octagon", command=self.draw_octagon).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Heart", command=self.draw_heart).pack(side=tk.LEFT)

        tk.Button(btn_frame, text="Verify Gauss's Law", command=self.check_gauss).pack(side=tk.LEFT)
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas).pack(side=tk.LEFT)

    def paint(self, event):
        x, y = event.x, event.y
        r = 2
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
        self.points.append([x, y])

    def draw_rectangle(self):
        self.canvas.create_rectangle(100, 100, 300, 300, outline="blue")
        rect_points = [[100,100],[300,100],[300,300],[100,300],[100,100]]
        self.points.extend(rect_points)

    def draw_circle(self):
        self.canvas.create_oval(150, 150, 350, 350, outline="red")
        circle_points = []
        for angle in np.linspace(0, 2*np.pi, 100):
            x = 250 + 100*np.cos(angle)
            y = 250 + 100*np.sin(angle)
            circle_points.append([x,y])
        circle_points.append(circle_points[0])
        self.points.extend(circle_points)

    def draw_oval(self):
        self.canvas.create_oval(120, 180, 380, 320, outline="purple")
        oval_points = []
        for angle in np.linspace(0, 2*np.pi, 100):
            x = 250 + 130*np.cos(angle)
            y = 250 + 70*np.sin(angle)
            oval_points.append([x,y])
        oval_points.append(oval_points[0])
        self.points.extend(oval_points)

    def draw_star(self):
        cx, cy, r = 250, 250, 100
        star_points = []
        for i in range(10):
            angle = i * np.pi/5
            radius = r if i % 2 == 0 else r/2
            x = cx + radius*np.cos(angle)
            y = cy + radius*np.sin(angle)
            star_points.append([x,y])
        star_points.append(star_points[0])
        self.canvas.create_polygon(star_points, outline="orange", fill="")
        self.points.extend(star_points)

    def draw_polygon(self, sides, radius=100, outline="black"):
        cx, cy = 250, 250
        poly_points = []
        for i in range(sides):
            angle = 2*np.pi*i/sides
            x = cx + radius*np.cos(angle)
            y = cy + radius*np.sin(angle)
            poly_points.append([x,y])
        poly_points.append(poly_points[0])
        self.canvas.create_polygon(poly_points, outline=outline, fill="")
        self.points.extend(poly_points)

    def draw_pentagon(self):
        self.draw_polygon(5, radius=100, outline="green")

    def draw_hexagon(self):
        self.draw_polygon(6, radius=100, outline="brown")

    def draw_octagon(self):
        self.draw_polygon(8, radius=100, outline="cyan")

    def draw_heart(self):
        heart_points = []
        for t in np.linspace(0, 2*np.pi, 200):
            x = 250 + 16*np.sin(t)**3 * 10
            y = 250 - (13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)) * 10
            heart_points.append([x,y])
        heart_points.append(heart_points[0])
        self.canvas.create_polygon(heart_points, outline="red", fill="")
        self.points.extend(heart_points)

    def check_gauss(self):
        if not self.points:
            messagebox.showinfo("Result", "No shape drawn.")
            return
        pts = np.array(self.points)
        verify_gauss(pts)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.points = []

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
