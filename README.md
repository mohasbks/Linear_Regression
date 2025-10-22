# 📊 Linear Regression from Scratch

A complete implementation of **Linear Regression** algorithm from scratch using **Gradient Descent** - without using Sklearn or any ML libraries!

## ✨ Features

- 🎯 **Pure Python Implementation** - Built from scratch using only NumPy
- 🌙 **Modern Dark Mode GUI** - Beautiful interactive interface using Tkinter
- 📈 **Real-time Visualization** - Live training progress with multiple plots
- 🔧 **Feature Scaling** - Automatic data normalization for better convergence
- 📊 **Multiple Datasets** - Pre-loaded examples (Height-Weight, House Prices, etc.)
- ⚙️ **Customizable Hyperparameters** - Adjust learning rate and iterations
- 📉 **Performance Metrics** - MSE, RMSE, and R² Score evaluation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/mohasbks/linear-regression-from-scratch.git
cd linear-regression-from-scratch

# Install dependencies
pip install -r requirements.txt
```

### Usage

#### 1. GUI Application (Recommended)
```bash
python linear_regression_gui.py
```

#### 2. Command Line Examples
```bash
python linear_regression_from_scratch.py
```

## 📸 Screenshots

### Dark Mode Interface
- Interactive dataset selection
- Real-time training visualization
- Multiple performance plots

### Training Results
- Regression line fitting
- Cost curve during training
- Residual distribution plot

## 🎓 How It Works

### The Algorithm

Linear Regression finds the best-fit line: **y = mx + b**

Where:
- `m` = slope (weight)
- `b` = intercept (bias)

### Gradient Descent

The algorithm uses **Gradient Descent** to minimize the cost function:

```
Cost = (1/2n) * Σ(y_pred - y_actual)²
```

**Update Rules:**
```
m = m - α * (1/n) * Σ(y_pred - y_actual) * x
b = b - α * (1/n) * Σ(y_pred - y_actual)
```

Where `α` is the learning rate.

### Feature Scaling

To prevent numerical overflow and improve convergence:
```
X_scaled = (X - mean(X)) / std(X)
```

## 📊 Available Datasets

1. **Height-Weight Relationship** - Predict weight based on height
2. **House Prices by Area** - Predict house prices based on area
3. **Simple Linear Data** - Basic linear relationship for testing
4. **Custom Data** - Randomly generated data

## 🔧 Hyperparameters

- **Learning Rate**: Controls the step size (default: 0.1)
- **Iterations**: Number of training epochs (default: 1000)

## 📈 Performance Metrics

- **MSE (Mean Squared Error)**: Average squared difference
- **RMSE (Root Mean Squared Error)**: Square root of MSE
- **R² Score**: Coefficient of determination (closer to 1 = better fit)

## 🎨 GUI Features

### Dark Mode Theme
- Modern dark interface with neon accents
- Easy on the eyes for long coding sessions
- Professional appearance

### Interactive Controls
- Dataset selection via radio buttons
- Adjustable hyperparameters
- One-click training
- Progress bar with status updates

### Visualizations
1. **Regression Fit Plot** - Data points with fitted line
2. **Cost Curve** - Training loss over iterations
3. **Residual Plot** - Error distribution analysis

## 📁 Project Structure

```
linear-regression-from-scratch/
├── linear_regression_from_scratch.py  # Core algorithm with examples
├── linear_regression_gui.py           # GUI application
├── requirements.txt                   # Dependencies
├── README.md                          # Documentation
└── .gitignore                         # Git ignore file
```

## 🛠️ Technical Details

### Dependencies
- `numpy` - Numerical computations
- `matplotlib` - Data visualization
- `tkinter` - GUI framework (built-in with Python)

### Key Components

**LinearRegressionFromScratch Class:**
- `fit()` - Train the model
- `predict()` - Make predictions
- `evaluate()` - Calculate performance metrics
- `gradient_descent()` - Update parameters
- `compute_cost()` - Calculate MSE

**LinearRegressionGUI Class:**
- Complete GUI implementation
- Multi-threaded training
- Real-time plot updates
- Dark mode styling

## 🎯 Learning Objectives

This project demonstrates:
- ✅ Understanding of Linear Regression mathematics
- ✅ Implementation of Gradient Descent from scratch
- ✅ Feature scaling and normalization
- ✅ Model evaluation metrics
- ✅ GUI development with Tkinter
- ✅ Data visualization with Matplotlib
- ✅ Multi-threading for responsive UI

## 🐛 Troubleshooting

### Model Diverging (Very Large Numbers)
- **Solution**: Feature scaling is now automatic
- Adjust learning rate if needed (try 0.01 - 0.5)

### Slow Training
- Reduce iterations or increase learning rate
- Training runs in background thread

### Import Errors
```bash
pip install --upgrade numpy matplotlib
```

## 📝 Example Results

```
╔══════════════════════════════════╗
║      TRAINING COMPLETED!         ║
╚══════════════════════════════════╝

📐 Final Equation:
   y = 0.9972x + -0.0000

📊 Performance Metrics:
   • MSE:  0.8031
   • RMSE: 0.8962
   • R² Score: 0.9944
   
✅ Excellent fit!
```

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

Built with ❤️ for learning and teaching Machine Learning fundamentals.

## 🌟 Acknowledgments

- Inspired by Andrew Ng's Machine Learning course
- Built without using Sklearn to understand the math
- Dark mode design for modern aesthetics

---

**⭐ If you found this helpful, please star the repository!**

## 📚 Further Reading

- [Gradient Descent Explained](https://en.wikipedia.org/wiki/Gradient_descent)
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression)
- [Feature Scaling Importance](https://en.wikipedia.org/wiki/Feature_scaling)
