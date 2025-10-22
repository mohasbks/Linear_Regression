"""
Linear Regression from Scratch - No Sklearn
Using Manual Gradient Descent

Equation: y = mx + b
Where:
- m: slope
- b: intercept
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegressionFromScratch:
    """
    Linear Regression algorithm from scratch using Gradient Descent
    """
    
    def __init__(self, learning_rate=0.01, iterations=1000):
        """
        Parameters:
        -----------
        learning_rate : float
            Learning rate for the algorithm
        iterations : int
            Number of iterations for training
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.m = 0  # slope
        self.b = 0  # intercept
        self.cost_history = []  # cost history for plotting
        
    def compute_cost(self, X, y):
        """
        Compute the cost (Mean Squared Error)
        Cost = (1/2n) * Σ(y_pred - y_actual)²
        """
        n = len(y)
        y_pred = self.m * X + self.b
        cost = (1/(2*n)) * np.sum((y_pred - y) ** 2)
        return cost
    
    def gradient_descent(self, X, y):
        """
        Apply Gradient Descent to update m and b
        
        Derivatives:
        - dm = (1/n) * Σ(y_pred - y_actual) * x
        - db = (1/n) * Σ(y_pred - y_actual)
        """
        n = len(y)
        y_pred = self.m * X + self.b
        
        # حساب المشتقات (gradients)
        dm = (1/n) * np.sum((y_pred - y) * X)
        db = (1/n) * np.sum(y_pred - y)
        
        # تحديث المعاملات
        self.m = self.m - self.learning_rate * dm
        self.b = self.b - self.learning_rate * db
        
    def fit(self, X, y):
        """
        Train the model on the data
        
        Parameters:
        -----------
        X : array-like
            Independent variable (e.g.: height)
        y : array-like
            Dependent variable (e.g.: weight)
        """
        # تحويل البيانات إلى numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Training with Gradient Descent
        for i in range(self.iterations):
            # Apply Gradient Descent
            self.gradient_descent(X, y)
            
            # Save cost every 10 iterations
            if i % 10 == 0:
                cost = self.compute_cost(X, y)
                self.cost_history.append(cost)
                
                # Print progress
                if i % 100 == 0:
                    print(f"Iteration {i}: Cost = {cost:.4f}, m = {self.m:.4f}, b = {self.b:.4f}")
        
        # Print final results
        final_cost = self.compute_cost(X, y)
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Final equation: y = {self.m:.4f}x + {self.b:.4f}")
        print(f"Final cost: {final_cost:.4f}")
        print(f"{'='*60}\n")
        
    def predict(self, X):
        """
        Predict new values
        
        Parameters:
        -----------
        X : array-like
            Values to predict
            
        Returns:
        --------
        predictions : array
            Predicted values
        """
        X = np.array(X)
        return self.m * X + self.b
    
    def plot_results(self, X, y, title="Linear Regression Results"):
        """
        Plot the results and data
        """
        plt.figure(figsize=(15, 5))
        
        # First plot: Data and regression line
        plt.subplot(1, 3, 1)
        plt.scatter(X, y, color='blue', alpha=0.6, s=50, label='Actual data')
        plt.plot(X, self.predict(X), color='red', linewidth=2, label=f'y = {self.m:.2f}x + {self.b:.2f}')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Second plot: Cost history
        plt.subplot(1, 3, 2)
        plt.plot(self.cost_history, color='green', linewidth=2)
        plt.xlabel('Iterations (×10)')
        plt.ylabel('Cost (MSE)')
        plt.title('Cost curve during training')
        plt.grid(True, alpha=0.3)
        
        # Third plot: Residuals
        plt.subplot(1, 3, 3)
        y_pred = self.predict(X)
        residuals = y - y_pred
        plt.scatter(X, residuals, color='purple', alpha=0.6, s=50)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
        plt.xlabel('X')
        plt.ylabel('Residuals (y_actual - y_pred)')
        plt.title('Residual distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def evaluate(self, X, y):
        """
        Evaluate model performance
        """
        X = np.array(X)
        y = np.array(y)
        
        y_pred = self.predict(X)
        
        # Mean Squared Error
        mse = np.mean((y - y_pred) ** 2)
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        
        # R-squared (معامل التحديد)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        print(f"Performance metrics:")
        print(f"- Mean Squared Error (MSE): {mse:.4f}")
        print(f"- Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"- R² Score: {r_squared:.4f}")
        print(f"  (Closer to 1 means better model)")


# ============================================
# مثال 1: العلاقة بين الطول والوزن
# ============================================
def example_height_weight():
    """
    Apply to height-weight relationship data
    """
    print("\n" + "="*60)
    print("Example 1: Height-Weight Relationship")
    print("="*60 + "\n")
    
    # Data: Height in centimeters
    heights = np.array([150, 155, 160, 165, 170, 175, 180, 185, 190, 195,
                       152, 158, 163, 168, 172, 178, 182, 187, 192, 197,
                       151, 156, 161, 166, 171, 176, 181, 186, 191, 196])
    
    # Weight in kilograms (with natural noise)
    weights = np.array([50, 53, 57, 60, 65, 68, 73, 77, 82, 87,
                       51, 55, 59, 63, 66, 71, 75, 79, 84, 89,
                       49, 54, 58, 61, 64, 69, 72, 78, 83, 88])
    
    # إنشاء النموذج والتدريب
    model = LinearRegressionFromScratch(learning_rate=0.0001, iterations=1000)
    model.fit(heights, weights)
    
    # تقييم النموذج
    model.evaluate(heights, weights)
    
    # Make predictions for new heights
    print(f"\nPrediction examples:")
    test_heights = [160, 170, 180, 190]
    for h in test_heights:
        predicted_weight = model.predict([h])[0]
        print(f"Height {h} cm → Predicted weight: {predicted_weight:.2f} kg")
    
    # Plot results
    model.plot_results(heights, weights, "Height-Weight Relationship")


# ============================================
# مثال 2: أسعار البيوت حسب المساحة
# ============================================
def example_house_prices():
    """
    Apply to house price data based on area
    """
    print("\n" + "="*60)
    print("Example 2: House Prices by Area")
    print("="*60 + "\n")
    
    # Data: Area in square meters
    areas = np.array([50, 60, 70, 80, 90, 100, 110, 120, 130, 140,
                     55, 65, 75, 85, 95, 105, 115, 125, 135, 145,
                     52, 62, 72, 82, 92, 102, 112, 122, 132, 142])
    
    # Price in thousands of dollars
    prices = np.array([150, 175, 200, 220, 245, 270, 290, 315, 340, 365,
                      160, 185, 210, 230, 250, 280, 300, 325, 345, 370,
                      155, 180, 205, 225, 248, 275, 295, 320, 342, 368])
    
    # إنشاء النموذج والتدريب
    model = LinearRegressionFromScratch(learning_rate=0.0001, iterations=1000)
    model.fit(areas, prices)
    
    # تقييم النموذج
    model.evaluate(areas, prices)
    
    # Make predictions for new areas
    print(f"\nPrediction examples:")
    test_areas = [60, 80, 100, 120, 150]
    for a in test_areas:
        predicted_price = model.predict([a])[0]
        print(f"Area {a} m² → Predicted price: ${predicted_price:.2f}k")
    
    # Plot results
    model.plot_results(areas, prices, "House Prices by Area")


# ============================================
# مثال 3: بيانات بسيطة للاختبار
# ============================================
def example_simple():
    """
    Simple example for testing
    """
    print("\n" + "="*60)
    print("Example 3: Simple Data (y = 2x + 3)")
    print("="*60 + "\n")
    
    # Simple data following y = 2x + 3 with small noise
    X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = 2 * X + 3 + np.random.randn(10) * 0.5  # Add random noise
    
    # إنشاء النموذج والتدريب
    model = LinearRegressionFromScratch(learning_rate=0.01, iterations=1000)
    model.fit(X, y)
    
    # تقييم النموذج
    model.evaluate(X, y)
    
    # Plot results
    model.plot_results(X, y, "Simple Example: y ≈ 2x + 3")


if __name__ == "__main__":
    print("="*60)
    print("Linear Regression from Scratch - No Sklearn")
    print("Using Manual Gradient Descent")
    print("="*60)
    
    # تشغيل جميع الأمثلة
    example_simple()
    example_height_weight()
    example_house_prices()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)
