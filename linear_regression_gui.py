"""
Linear Regression GUI - Interactive Application
Complete GUI with data visualization and real-time training
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox
import threading


class LinearRegressionFromScratch:
    """Linear Regression algorithm from scratch using Gradient Descent"""
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0.0  # slope
        self.b = 0.0  # intercept
        self.cost_history = []
        
        # For feature scaling
        self.X_mean = 0
        self.X_std = 1
        self.y_mean = 0
        self.y_std = 1
        
    def compute_cost(self, X, y):
        """Compute the cost (Mean Squared Error)"""
        n = len(y)
        y_pred = self.m * X + self.b
        cost = (1/(2*n)) * np.sum((y_pred - y) ** 2)
        # Handle overflow
        if np.isnan(cost) or np.isinf(cost):
            return 1e10
        return cost
    
    def gradient_descent(self, X, y):
        """Apply Gradient Descent to update m and b"""
        n = len(y)
        y_pred = self.m * X + self.b
        
        # Calculate gradients
        dm = (1/n) * np.sum((y_pred - y) * X)
        db = (1/n) * np.sum(y_pred - y)
        
        # Clip gradients to prevent overflow
        dm = np.clip(dm, -1e10, 1e10)
        db = np.clip(db, -1e10, 1e10)
        
        # Update parameters
        new_m = self.m - self.learning_rate * dm
        new_b = self.b - self.learning_rate * db
        
        # Check for NaN or Inf
        if not (np.isnan(new_m) or np.isinf(new_m)):
            self.m = new_m
        if not (np.isnan(new_b) or np.isinf(new_b)):
            self.b = new_b
        
    def fit(self, X, y, callback=None):
        """Train the model on the data with optional callback for GUI updates"""
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        
        # Feature Scaling (Standardization)
        self.X_mean = np.mean(X)
        self.X_std = np.std(X)
        self.y_mean = np.mean(y)
        self.y_std = np.std(y)
        
        # Avoid division by zero
        if self.X_std == 0:
            self.X_std = 1
        if self.y_std == 0:
            self.y_std = 1
        
        # Normalize data
        X_scaled = (X - self.X_mean) / self.X_std
        y_scaled = (y - self.y_mean) / self.y_std
        
        for i in range(self.iterations):
            self.gradient_descent(X_scaled, y_scaled)
            
            if i % 10 == 0:
                cost = self.compute_cost(X_scaled, y_scaled)
                self.cost_history.append(cost)
                
                # Call callback for GUI updates
                if callback and i % 50 == 0:
                    callback(i, cost, self.m, self.b)
        
        final_cost = self.compute_cost(X_scaled, y_scaled)
        return final_cost
        
    def predict(self, X):
        """Predict new values"""
        X = np.array(X, dtype=np.float64)
        # Scale input
        X_scaled = (X - self.X_mean) / self.X_std
        # Predict on scaled data
        y_scaled = self.m * X_scaled + self.b
        # Unscale output
        y = y_scaled * self.y_std + self.y_mean
        return y
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        y_pred = self.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        return mse, rmse, r_squared


class LinearRegressionGUI:
    """Modern GUI for Linear Regression"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression from Scratch - Dark Mode GUI")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # Dark mode color scheme
        self.bg_dark = '#1e1e1e'
        self.bg_panel = '#2d2d2d'
        self.bg_widget = '#3d3d3d'
        self.fg_primary = '#ffffff'
        self.fg_secondary = '#b0b0b0'
        self.accent_blue = '#0d7377'
        self.accent_green = '#14ffec'
        self.accent_red = '#ff6b6b'
        self.accent_purple = '#a78bfa'
        self.accent_orange = '#fb923c'
        
        # Data storage
        self.X_data = None
        self.y_data = None
        self.model = None
        self.is_training = False
        
        # Create GUI components
        self.create_widgets()
        
    def create_widgets(self):
        """Create all GUI widgets"""
        
        # Main container
        main_frame = tk.Frame(self.root, bg=self.bg_dark)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # ===== LEFT PANEL: Controls =====
        left_panel = tk.Frame(main_frame, bg=self.bg_panel, relief=tk.RAISED, borderwidth=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), pady=0)
        
        # Title
        title_label = tk.Label(left_panel, text="🎯 Linear Regression", 
                              font=('Arial', 20, 'bold'), bg=self.bg_panel, fg=self.accent_green)
        title_label.pack(pady=20)
        
        # Dataset Selection
        dataset_frame = tk.LabelFrame(left_panel, text="📊 Select Dataset", 
                                     font=('Arial', 12, 'bold'), bg=self.bg_panel, 
                                     fg=self.fg_primary, borderwidth=2, relief=tk.GROOVE)
        dataset_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.dataset_var = tk.StringVar(value="height_weight")
        datasets = [
            ("Height-Weight Relationship", "height_weight"),
            ("House Prices by Area", "house_prices"),
            ("Simple Linear Data", "simple"),
            ("Custom Data", "custom")
        ]
        
        for text, value in datasets:
            rb = tk.Radiobutton(dataset_frame, text=text, variable=self.dataset_var, 
                               value=value, font=('Arial', 10), bg=self.bg_panel,
                               fg=self.fg_primary, selectcolor=self.bg_widget,
                               activebackground=self.bg_panel, activeforeground=self.accent_green,
                               command=self.load_dataset)
            rb.pack(anchor=tk.W, padx=10, pady=5)
        
        # Hyperparameters
        params_frame = tk.LabelFrame(left_panel, text="⚙️ Hyperparameters", 
                                    font=('Arial', 12, 'bold'), bg=self.bg_panel, 
                                    fg=self.fg_primary, borderwidth=2, relief=tk.GROOVE)
        params_frame.pack(fill=tk.X, padx=20, pady=10)
        
        # Learning Rate
        tk.Label(params_frame, text="Learning Rate:", font=('Arial', 10), 
                bg=self.bg_panel, fg=self.fg_secondary).grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)
        self.lr_var = tk.StringVar(value="0.1")
        lr_entry = tk.Entry(params_frame, textvariable=self.lr_var, width=15, font=('Arial', 10),
                           bg=self.bg_widget, fg=self.fg_primary, insertbackground=self.fg_primary)
        lr_entry.grid(row=0, column=1, padx=10, pady=5)
        
        # Iterations
        tk.Label(params_frame, text="Iterations:", font=('Arial', 10), 
                bg=self.bg_panel, fg=self.fg_secondary).grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)
        self.iter_var = tk.StringVar(value="1000")
        iter_entry = tk.Entry(params_frame, textvariable=self.iter_var, width=15, font=('Arial', 10),
                             bg=self.bg_widget, fg=self.fg_primary, insertbackground=self.fg_primary)
        iter_entry.grid(row=1, column=1, padx=10, pady=5)
        
        # Train Button
        self.train_button = tk.Button(left_panel, text="🚀 Train Model", 
                                     font=('Arial', 14, 'bold'), bg=self.accent_blue, fg=self.fg_primary,
                                     command=self.train_model, cursor='hand2',
                                     activebackground=self.accent_green, activeforeground=self.bg_dark,
                                     relief=tk.RAISED, borderwidth=3, padx=20, pady=10)
        self.train_button.pack(pady=20)
        
        # Results Display
        results_frame = tk.LabelFrame(left_panel, text="📈 Results", 
                                     font=('Arial', 12, 'bold'), bg=self.bg_panel, 
                                     fg=self.fg_primary, borderwidth=2, relief=tk.GROOVE)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.results_text = tk.Text(results_frame, height=15, width=35, 
                                   font=('Courier', 9), bg=self.bg_widget, fg=self.fg_primary,
                                   insertbackground=self.fg_primary,
                                   relief=tk.SUNKEN, borderwidth=2)
        self.results_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        # Progress Bar
        self.progress = ttk.Progressbar(left_panel, mode='determinate', length=300)
        self.progress.pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(left_panel, text="Ready to train", 
                                    font=('Arial', 10), bg=self.bg_panel, fg=self.accent_green)
        self.status_label.pack(pady=5)
        
        # ===== RIGHT PANEL: Visualizations =====
        right_panel = tk.Frame(main_frame, bg=self.bg_panel, relief=tk.RAISED, borderwidth=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(12, 8), facecolor=self.bg_panel)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Initialize with default dataset
        self.load_dataset()
        
    def load_dataset(self):
        """Load selected dataset"""
        dataset = self.dataset_var.get()
        
        if dataset == "height_weight":
            # Height-Weight data
            self.X_data = np.array([150, 155, 160, 165, 170, 175, 180, 185, 190, 195,
                                   152, 158, 163, 168, 172, 178, 182, 187, 192, 197,
                                   151, 156, 161, 166, 171, 176, 181, 186, 191, 196])
            self.y_data = np.array([50, 53, 57, 60, 65, 68, 73, 77, 82, 87,
                                   51, 55, 59, 63, 66, 71, 75, 79, 84, 89,
                                   49, 54, 58, 61, 64, 69, 72, 78, 83, 88])
            self.x_label = "Height (cm)"
            self.y_label = "Weight (kg)"
            self.lr_var.set("0.1")
            
        elif dataset == "house_prices":
            # House prices data
            self.X_data = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                                   55, 65, 75, 85, 95, 105, 115, 125, 135, 145,
                                   52, 62, 72, 82, 92, 102, 112, 122, 132, 142])
            self.y_data = np.array([150, 175, 200, 220, 245, 270, 290, 315, 340, 365,
                                   160, 185, 210, 230, 250, 280, 300, 325, 345, 370,
                                   155, 180, 205, 225, 248, 275, 295, 320, 342, 368])
            self.x_label = "Area (m²)"
            self.y_label = "Price ($k)"
            self.lr_var.set("0.1")
            
        elif dataset == "simple":
            # Simple linear data
            self.X_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
            self.y_data = 2 * self.X_data + 3 + np.random.randn(10) * 0.5
            self.x_label = "X"
            self.y_label = "y"
            self.lr_var.set("0.1")
            
        elif dataset == "custom":
            # Generate random data
            self.X_data = np.linspace(0, 100, 30)
            self.y_data = 2.5 * self.X_data + 10 + np.random.randn(30) * 10
            self.x_label = "X"
            self.y_label = "y"
            self.lr_var.set("0.1")
        
        # Clear results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Dataset loaded: {dataset}\n")
        self.results_text.insert(tk.END, f"Data points: {len(self.X_data)}\n")
        self.results_text.insert(tk.END, f"\nReady to train!\n")
        
        # Plot initial data
        self.plot_initial_data()
        
    def plot_initial_data(self):
        """Plot the initial dataset"""
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        
        # Dark theme styling
        ax.set_facecolor(self.bg_widget)
        ax.scatter(self.X_data, self.y_data, color=self.accent_green, alpha=0.7, s=100, 
                  edgecolors=self.accent_blue, linewidth=1.5, label='Data Points')
        ax.set_xlabel(self.x_label, fontsize=12, fontweight='bold', color=self.fg_primary)
        ax.set_ylabel(self.y_label, fontsize=12, fontweight='bold', color=self.fg_primary)
        ax.set_title('Dataset Visualization', fontsize=14, fontweight='bold', pad=20, color=self.accent_green)
        ax.grid(True, alpha=0.2, linestyle='--', color=self.fg_secondary)
        ax.tick_params(colors=self.fg_secondary)
        ax.spines['bottom'].set_color(self.fg_secondary)
        ax.spines['top'].set_color(self.fg_secondary)
        ax.spines['left'].set_color(self.fg_secondary)
        ax.spines['right'].set_color(self.fg_secondary)
        legend = ax.legend(fontsize=10)
        legend.get_frame().set_facecolor(self.bg_panel)
        legend.get_frame().set_edgecolor(self.fg_secondary)
        for text in legend.get_texts():
            text.set_color(self.fg_primary)
        
        self.canvas.draw()
        
    def train_model(self):
        """Train the model in a separate thread"""
        if self.is_training:
            messagebox.showwarning("Training", "Model is already training!")
            return
        
        try:
            learning_rate = float(self.lr_var.get())
            iterations = int(self.iter_var.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid hyperparameters!")
            return
        
        self.is_training = True
        self.train_button.config(state=tk.DISABLED, bg=self.bg_widget)
        self.status_label.config(text="Training in progress...", fg=self.accent_orange)
        self.progress['value'] = 0
        
        # Run training in separate thread
        thread = threading.Thread(target=self._train_thread, 
                                 args=(learning_rate, iterations))
        thread.daemon = True
        thread.start()
        
    def _train_thread(self, learning_rate, iterations):
        """Training thread"""
        self.model = LinearRegressionFromScratch(learning_rate, iterations)
        
        def update_callback(iteration, cost, m, b):
            progress = (iteration / iterations) * 100
            self.root.after(0, lambda: self.progress.config(value=progress))
            
        # Train model
        final_cost = self.model.fit(self.X_data, self.y_data, update_callback)
        
        # Evaluate
        mse, rmse, r_squared = self.model.evaluate(self.X_data, self.y_data)
        
        # Update GUI
        self.root.after(0, lambda: self.update_results(final_cost, mse, rmse, r_squared))
        self.root.after(0, self.plot_results)
        self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL, bg=self.accent_blue))
        self.root.after(0, lambda: self.status_label.config(text="Training completed!", fg=self.accent_green))
        self.root.after(0, lambda: self.progress.config(value=100))
        
        self.is_training = False
        
    def update_results(self, final_cost, mse, rmse, r_squared):
        """Update results text"""
        self.results_text.delete(1.0, tk.END)
        
        results = f"""
╔══════════════════════════════════╗
║      TRAINING COMPLETED!         ║
╚══════════════════════════════════╝

📐 Final Equation:
   y = {self.model.m:.4f}x + {self.model.b:.4f}

📊 Performance Metrics:
   • MSE:  {mse:.4f}
   • RMSE: {rmse:.4f}
   • R² Score: {r_squared:.4f}
   
   (R² closer to 1 = better fit)

⚙️ Training Details:
   • Final Cost: {final_cost:.4f}
   • Learning Rate: {self.model.learning_rate}
   • Iterations: {self.model.iterations}
   • Data Points: {len(self.X_data)}

✅ Model is ready for predictions!
"""
        self.results_text.insert(tk.END, results)
        
    def plot_results(self):
        """Plot training results with dark theme"""
        self.fig.clear()
        
        # Create 2x2 subplot
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Data and Regression Line
        ax1 = self.fig.add_subplot(gs[0, :])
        ax1.set_facecolor(self.bg_widget)
        ax1.scatter(self.X_data, self.y_data, color=self.accent_green, alpha=0.7, s=100, 
                   edgecolors=self.accent_blue, linewidth=1.5, label='Actual Data')
        
        # Plot regression line
        x_line = np.linspace(self.X_data.min(), self.X_data.max(), 100)
        y_line = self.model.predict(x_line)
        ax1.plot(x_line, y_line, color=self.accent_red, linewidth=3, 
                label=f'y = {self.model.m:.2f}x + {self.model.b:.2f}')
        
        ax1.set_xlabel(self.x_label, fontsize=11, fontweight='bold', color=self.fg_primary)
        ax1.set_ylabel(self.y_label, fontsize=11, fontweight='bold', color=self.fg_primary)
        ax1.set_title('Linear Regression Fit', fontsize=13, fontweight='bold', pad=15, color=self.accent_green)
        ax1.tick_params(colors=self.fg_secondary)
        self._style_dark_axes(ax1)
        
        # 2. Cost History
        ax2 = self.fig.add_subplot(gs[1, 0])
        ax2.set_facecolor(self.bg_widget)
        ax2.plot(self.model.cost_history, color=self.accent_green, linewidth=2.5)
        ax2.set_xlabel('Iterations (×10)', fontsize=10, fontweight='bold', color=self.fg_primary)
        ax2.set_ylabel('Cost (MSE)', fontsize=10, fontweight='bold', color=self.fg_primary)
        ax2.set_title('Training Cost Curve', fontsize=12, fontweight='bold', pad=10, color=self.accent_green)
        ax2.tick_params(colors=self.fg_secondary)
        ax2.fill_between(range(len(self.model.cost_history)), 
                        self.model.cost_history, alpha=0.3, color=self.accent_green)
        self._style_dark_axes(ax2)
        
        # 3. Residuals
        ax3 = self.fig.add_subplot(gs[1, 1])
        ax3.set_facecolor(self.bg_widget)
        y_pred = self.model.predict(self.X_data)
        residuals = self.y_data - y_pred
        ax3.scatter(self.X_data, residuals, color=self.accent_purple, alpha=0.7, s=80,
                   edgecolors=self.accent_blue, linewidth=1.5)
        ax3.axhline(y=0, color=self.accent_red, linestyle='--', linewidth=2.5)
        ax3.set_xlabel(self.x_label, fontsize=10, fontweight='bold', color=self.fg_primary)
        ax3.set_ylabel('Residuals', fontsize=10, fontweight='bold', color=self.fg_primary)
        ax3.set_title('Residual Plot', fontsize=12, fontweight='bold', pad=10, color=self.accent_green)
        ax3.tick_params(colors=self.fg_secondary)
        self._style_dark_axes(ax3)
        
        self.canvas.draw()
    
    def _style_dark_axes(self, ax):
        """Apply dark theme styling to axes"""
        ax.grid(True, alpha=0.2, linestyle='--', color=self.fg_secondary)
        ax.spines['bottom'].set_color(self.fg_secondary)
        ax.spines['top'].set_color(self.fg_secondary)
        ax.spines['left'].set_color(self.fg_secondary)
        ax.spines['right'].set_color(self.fg_secondary)
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(fontsize=10, loc='best')
            legend.get_frame().set_facecolor(self.bg_panel)
            legend.get_frame().set_edgecolor(self.fg_secondary)
            for text in legend.get_texts():
                text.set_color(self.fg_primary)


def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = LinearRegressionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
