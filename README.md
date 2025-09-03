CS5710 Machine Learning – Homework 1
====================================

*Student Name:* AJAY MUPPA  
*Student ID:* 700769264

Overview
--------
This homework covers seven tasks, from manual function approximation and cost evaluation
to implementing gradient descent for linear regression. It demonstrates understanding of
error measurement, optimization, underfitting/overfitting, and model evaluation.

Tasks Summary
-------------
1. Function Approximation by Hand:
   - Tested θ=(1,0) and θ=(0.5,1) on dataset.
   - MSEs: 4.5 vs 0.75. Model (0.5,1) fits better.

2. Random Guessing Practice:
   - Cost J evaluated at (0.1,0.2)=1.32, (0.5,0.9)=0.48.
   - (0.5,0.9) is closer to the minimum (0.3,0.7).
   - Random guessing is inefficient because it ignores gradient information.

3. First Gradient Descent Iteration:
   - Dataset: (1,3),(2,4),(3,6),(4,5).
   - Step 1: θ updated from (0,0) to (0.09,0.245), cost decreased 21.5→15.3.
   - Step 2: θ updated to (0.166,0.449), cost decreased further to 10.9.

4. Random Guessing vs Gradient Descent:
   - Random guesses gave MSEs: 5.5 and 7.9.
   - One GD step gave MSE ~10.5.
   - Random can be better at first, but GD improves systematically.

5. Recognizing Underfitting:
   - High training and test errors → underfitting.
   - Fixes: increase complexity, improve features, reduce regularization.

6. Comparing Models:
   - Model A: overfits (low bias, high variance). Fix with regularization/simplification.
   - Model B: underfits (high bias). Fix with more features/complexity.

7. Programming: Gradient Descent vs Closed-Form Linear Regression:
   - Synthetic data y = 3 + 4x + ε, n=200.
   - Closed-form solution: θ0≈2.69, θ1≈4.13.
   - Gradient Descent (η=0.05, 1000 iters): θ0≈2.69, θ1≈4.13.
   - GD converges to closed-form solution.
   - Outputs: task7_dataset.csv, task7_linear_regression.py, task7_fits.png, task7_loss.png.

Deliverables
------------
- Solutions explained in Word/PDF and presentation format.
- Code, dataset, plots, and README provided for Task 7.

Conclusion
----------
This homework highlights the importance of systematic optimization methods over random guessing,
demonstrates the progression of gradient descent, and distinguishes between underfitting and overfitting.
The coding task confirms that gradient descent converges to the closed-form solution.
