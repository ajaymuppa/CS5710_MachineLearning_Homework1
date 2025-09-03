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
# Task 1 — Function approximation by hand
**Dataset:** $(x,y)=\{(1,1),(2,2),(3,2),(4,5)\}$.

We evaluate two models and compute full arithmetic for residuals and MSE.

### Model A: $\theta=(1,0)$, so $\hat{y}=1+0\cdot x = 1$ for every $x$.

We build a table and show how each column is computed.

| x | y | $\hat{y}$ | residual $r=\hat{y}-y$ | $r^2$ |
|---:|---:|---:|---:|---:|
| 1 | 1 | 1 | $1-1 = 0$ | $0^2 = 0$ |
| 2 | 2 | 1 | $1-2 = -1$ | $(-1)^2 = 1$ |
| 3 | 2 | 1 | $1-2 = -1$ | $1$ |
| 4 | 5 | 1 | $1-5 = -4$ | $(-4)^2 = 16$ |

**Step-by-step arithmetic:**
- For x=1: prediction $\hat y=1$, residual $0$, square $0$.
- For x=2: prediction $\hat y=1$, residual $-1$, square $1$.
- For x=3: residual $-1$, square $1$.
- For x=4: residual $-4$, square $16$.

Sum of squared residuals: $0 + 1 + 1 + 16 = 18$.  
Number of points $n=4$.  
MSE $=18/4 = \boxed{4.5}$.

---

### Model B: $\theta=(0.5,1)$, so $\hat{y} = 0.5 + 1\cdot x = x + 0.5$.

Table:

| x | y | $\hat{y}=x+0.5$ | residual $r=\hat{y}-y$ | $r^2$ |
|---:|---:|---:|---:|---:|
| 1 | 1 | $1 + 0.5 = 1.5$ | $1.5-1 = 0.5$ | $0.5^2 = 0.25$ |
| 2 | 2 | $2 + 0.5 = 2.5$ | $2.5-2 = 0.5$ | $0.25$ |
| 3 | 2 | $3 + 0.5 = 3.5$ | $3.5-2 = 1.5$ | $1.5^2 = 2.25$ |
| 4 | 5 | $4 + 0.5 = 4.5$ | $4.5-5 = -0.5$ | $(-0.5)^2 = 0.25$ |

Sum of squared residuals: $0.25 + 0.25 + 2.25 + 0.25 = 3.0$.  
MSE $=3.0 / 4 = \boxed{0.75}$.

**Conclusion (Task 1):** Model B ($\theta=(0.5,1)$) is the better fit because it has the lower MSE (0.75 vs 4.5).

---

# Task 2 — Random guessing practice (evaluate a given cost function)
We are given the convex cost:
\[
\boxed{J(\theta_0,\theta_1)=8(\theta_0-0.3)^2 + 4(\theta_1-0.7)^2.}
\]
The minimum is at $(\theta_0,\theta_1)=(0.3,0.7)$, but we evaluate two guesses.

### Evaluate $J(0.1,0.2)$
Compute each term separately:

- For $\theta_0$-term: $\theta_0 - 0.3 = 0.1 - 0.3 = -0.2$. Square: $(-0.2)^2 = 0.04$. Multiply by 8: $8 \cdot 0.04 = 0.32$.
- For $\theta_1$-term: $\theta_1 - 0.7 = 0.2 - 0.7 = -0.5$. Square: $(-0.5)^2 = 0.25$. Multiply by 4: $4 \cdot 0.25 = 1.00$.

Sum: $0.32 + 1.00 = \boxed{1.32}$.

### Evaluate $J(0.5,0.9)$
- $\theta_0$-term: $0.5 - 0.3 = 0.2 \Rightarrow 0.2^2 = 0.04 \Rightarrow 8\cdot0.04 = 0.32$.
- $\theta_1$-term: $0.9 - 0.7 = 0.2 \Rightarrow 0.2^2 = 0.04 \Rightarrow 4\cdot0.04 = 0.16$.

Sum: $0.32 + 0.16 = \boxed{0.48}$.

**Which is closer to the minimum?** $(0.5,0.9)$ has the smaller $J$ (0.48) than $(0.1,0.2)$ (1.32), so $(0.5,0.9)$ is closer.

**Short reasoning (why random guessing is inefficient):** Random guesses in continuous parameter spaces rarely land near a narrow minimum; gradient-based methods use local slope information to move directly toward the minimum instead of searching blindly.

---

# Task 3 — First gradient descent iteration (with full arithmetic)
**Dataset:** $(1,3),(2,4),(3,6),(4,5)$.  
Start: $\theta^{(0)}=(0,0)$. Learning rate $\alpha = 0.01$. Use MSE: $J=\frac{1}{n}\sum r_i^2$.

### Step A — Compute residuals at $\theta^{(0)}=(0,0)$
Predictions $\hat y_i = 0$ (since both parameters are 0). Residuals $r_i = \hat y_i - y_i = 0 - y_i = -y_i$:

- For (1,3): $r_1 = -3$.
- For (2,4): $r_2 = -4$.
- For (3,6): $r_3 = -6$.
- For (4,5): $r_4 = -5$.

Compute the sums needed for the gradient:

- $\sum r_i = -3 + (-4) + (-6) + (-5) = -18$.
- $\sum x_i r_i = 1\cdot(-3) + 2\cdot(-4) + 3\cdot(-6) + 4\cdot(-5)$  
  $= -3 - 8 - 18 - 20 = -49$.

The gradient (using $\frac{2}{n}$ factor) is:
\[
\nabla J(\theta^{(0)}) = \frac{2}{4}\begin{bmatrix}-18\\-49\end{bmatrix} = \begin{bmatrix}-9\\-24.5\end{bmatrix}.
\]

### Step B — Gradient descent update (first iteration)
\[
\theta^{(1)} = \theta^{(0)} - \alpha \nabla J(\theta^{(0)})
= (0,0) - 0.01\cdot(-9,\ -24.5)
= (\;0.09,\ 0.245\;).
\]
So $\boxed{\theta^{(1)}=(0.09,\ 0.245)}$.

### Step C — Compute MSE at $\theta^{(0)}$ (initial) and $\theta^{(1)}$

**Initial MSE $J(\theta^{(0)})$**:
Squares of residuals: $(-3)^2=9, \; (-4)^2=16,\; (-6)^2=36,\; (-5)^2=25$. Sum $=9+16+36+25=86$.  
$J(\theta^{(0)}) = 86 / 4 = \boxed{21.5}$.

**MSE at $\theta^{(1)}=(0.09,0.245)$**  
First compute predictions $\hat y = 0.09 + 0.245 x$ and residuals:

- x=1: $\hat y = 0.09 + 0.245(1) = 0.335$. Residual $r = 0.335 - 3 = -2.665$. Square $r^2 = 2.665^2 = 7.102225$.
- x=2: $\hat y = 0.09 + 0.245(2) = 0.09 + 0.49 = 0.58$. Residual $r = 0.58 - 4 = -3.42$. Square $=3.42^2 = 11.6964$.
- x=3: $\hat y = 0.09 + 0.245(3) = 0.09 + 0.735 = 0.825$. Residual $= 0.825 -6 = -5.175$. Square $=5.175^2 = 26.780625$.
- x=4: $\hat y = 0.09 + 0.245(4) = 0.09 + 0.98 = 1.07$. Residual $= 1.07 - 5 = -3.93$. Square $=3.93^2 = 15.4449$.

Sum of squared residuals $=7.102225 + 11.6964 + 26.780625 + 15.4449 = 61.02415$.  
MSE $= 61.02415 / 4 = \boxed{15.2560375}$.

**Observation:** One GD step reduced MSE from $21.5$ to approximately $15.2560$.

---

### (Optional — we continue one more GD iteration here and show steps)
**Why continue?** The user asked for step-by-step arithmetic and to "explain the tables or solve them" — showing a second iteration helps illustrate the pattern.

At $\theta^{(1)}$: we already computed the residuals (they sum to $-15.19$ when added):

- Residuals: $r_1=-2.665$, $r_2=-3.42$, $r_3=-5.175$, $r_4=-3.93$.
- $\sum r = -2.665 - 3.42 - 5.175 - 3.93 = -15.19$.
- $\sum x r = 1(-2.665) + 2(-3.42) + 3(-5.175) + 4(-3.93)$  
  $= -2.665 - 6.84 - 15.525 - 15.72 = -40.75$.

Gradient:
\[
\nabla J(\theta^{(1)}) = \frac{2}{4}\begin{bmatrix}-15.19\\-40.75\end{bmatrix}
= \begin{bmatrix}-7.595\\-20.375\end{bmatrix}.
\]

Update:
\[
\theta^{(2)} = \theta^{(1)} - 0.01\cdot \nabla J(\theta^{(1)})
= (0.09,\ 0.245) - 0.01\cdot(-7.595,\ -20.375)
= (0.09 + 0.07595,\ 0.245 + 0.20375)
= \boxed{(0.16595,\ 0.44875)}.
\]

Now compute MSE at $\theta^{(2)}$:

Predictions $\hat y = 0.16595 + 0.44875 x$:

- x=1: $\hat y = 0.16595 + 0.44875 = 0.6147$. Residual $=0.6147 - 3 = -2.3853$. Square $=2.3853^2 = 5.68965609$.
- x=2: $\hat y = 0.16595 + 2\cdot0.44875 = 0.16595 + 0.8975 = 1.06345$. Residual $=1.06345 - 4 = -2.93655$. Square $=2.93655^2 = 8.6233259025$.
- x=3: $\hat y = 0.16595 + 3\cdot0.44875 = 1.51220$. Residual $=1.51220 - 6 = -4.48780$. Square $=4.4878^2 = 20.14034884$.
- x=4: $\hat y = 0.16595 + 4\cdot0.44875 = 1.96095$. Residual $=1.96095 - 5 = -3.03905$. Square $=3.03905^2 = 9.2358249025$.

Sum squares $=5.68965609 + 8.6233259025 + 20.14034884 + 9.2358249025 = 43.689155735$.  
MSE $=43.689155735/4 = \boxed{10.92228893375}$.

**Comparison:** MSE decreased again: $21.5 \rightarrow 15.2560 \rightarrow 10.9223$.

---

# Task 4 — Random guessing vs gradient descent (full arithmetic)
**Dataset:** $(1,2),(2,2),(3,4),(4,6)$.

We compare two random guesses and one first GD step (starting at $\theta=(0,0)$) — computing full tables and MSEs.

---

### Random guess 1: $\theta=(0.2,0.5)$, so $\hat{y}=0.2 + 0.5x$.

Compute predictions, residuals, squares:

| x | y | $\hat{y}=0.2+0.5x$ | residual $r=\hat{y}-y$ | $r^2$ |
|---:|---:|---:|---:|---:|
| 1 | 2 | $0.2+0.5(1)=0.7$ | $0.7-2 = -1.3$ | $(-1.3)^2 = 1.69$ |
| 2 | 2 | $0.2+0.5(2)=1.2$ | $1.2-2 = -0.8$ | $0.64$ |
| 3 | 4 | $0.2+0.5(3)=1.7$ | $1.7-4 = -2.3$ | $5.29$ |
| 4 | 6 | $0.2+0.5(4)=2.2$ | $2.2-6 = -3.8$ | $14.44$ |

Sum squares $=1.69 + 0.64 + 5.29 + 14.44 = 22.06$.  
MSE $=22.06/4 = \boxed{5.515}$.

---

### Random guess 2: $\theta=(0.9,0.1)$, so $\hat{y}=0.9 + 0.1x$.

| x | y | $\hat{y}$ | residual | $r^2$ |
|---:|---:|---:|---:|---:|
|1|2|$0.9+0.1(1)=1.0$|$1.0-2=-1.0$|$1.00$|
|2|2|$0.9+0.1(2)=1.1$|$1.1-2=-0.9$|$0.81$|
|3|4|$0.9+0.1(3)=1.2$|$1.2-4=-2.8$|$7.84$|
|4|6|$0.9+0.1(4)=1.3$|$1.3-6=-4.7$|$22.09$|

Sum squares $=1 + 0.81 + 7.84 + 22.09 = 31.74$.  
MSE $=31.74 / 4 = \boxed{7.935}$.

---

### First GD step from $\theta^{(0)}=(0,0)$ (with $\alpha = 0.01$)

Residuals at $\theta^{(0)}$ are $r_i = 0 - y_i = -y_i$, so for the points:
- $r = [-2, -2, -4, -6]$.

Compute sums:
- $\sum r = -2 -2 -4 -6 = -14$.
- $\sum x r = 1(-2) + 2(-2) + 3(-4) + 4(-6) = -2 -4 -12 -24 = -42$.

Gradient: $\nabla J = \frac{2}{4}[-14,-42] = [-7,-21]$.

Update:
\[
\theta^{(1)} = (0,0) - 0.01\cdot(-7, -21) = (0.07,\ 0.21).
\]

Now compute MSE at $\theta^{(1)}$:

Predictions $\hat{y} = 0.07 + 0.21 x$:

| x | y | $\hat{y}$ | residual $r$ | $r^2$ |
|---:|---:|---:|---:|---:|
|1|2|$0.07+0.21(1)=0.28$|$0.28-2=-1.72$|$(-1.72)^2=2.9584$|
|2|2|$0.07+0.21(2)=0.49$|$0.49-2=-1.51$|$2.2801$|
|3|4|$0.07+0.21(3)=0.70$|$0.70-4=-3.30$|$10.89$|
|4|6|$0.07+0.21(4)=0.91$|$0.91-6=-5.09$|$25.9081$|

Sum squares $=2.9584 + 2.2801 + 10.89 + 25.9081 = 42.0366$.  
MSE $=42.0366 / 4 = \boxed{10.50915}$.

---

**Which had the lower error?** Among the three MSE values:
- Random guess (0.2,0.5): **5.515** (best here).
- Random guess (0.9,0.1): 7.935.
- First GD step (0.07,0.21): 10.50915.

A single small GD step does not always beat a lucky random guess; GD is reliable and systematic and will reduce error over many iterations, while a random guess can occasionally be closer by chance.

---

# Task 5 — Recognizing underfitting vs overfitting
**Given statement:** Training error is very high and test error is very high.

**Answer:** This indicates **underfitting** (high bias). The model is too simple to capture the structure in the training data, so it performs poorly both on train and test.

**Two ways to fix underfitting:**
1. Increase model capacity: use a more flexible model (e.g., add polynomial features, increase network size).
2. Improve features: engineer better input features or include more informative variables; reduce overly-strong regularization.

---

# Task 6 — Comparing models A and B
- **Model A:** "Trains almost perfectly but poor on new data."  
  - Diagnosis: **Overfitting** (low bias, high variance).  
  - Remedies: Increase regularization (e.g., larger weight penalty), reduce model complexity, collect more training data, use data augmentation, or use early stopping.

- **Model B:** "Poor on both train and test sets."  
  - Diagnosis: **Underfitting** (high bias).  
  - Remedies: Increase model capacity, add or engineer better features, reduce regularization strength, or train longer / tune hyperparameters.

---

# Task 7 — Gradient Descent vs Closed-Form Linear Regression
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
