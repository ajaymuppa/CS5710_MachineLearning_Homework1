Task 7: Linear Regression via Closed-form and Gradient Descent
Dataset: y = 3 + 4x + ε with Gaussian noise, n=200, x in [0,5].
Closed-form solution uses the normal equation on [1, x].
Gradient Descent uses lr=0.05 for 1000 iters, starting at θ=[0,0].
Outputs: task7_fits.png (data and two fitted lines), task7_loss.png (MSE vs iterations).
Both methods recover parameters close to the ground truth [3, 4]—GD converges to the closed-form within tolerance.
