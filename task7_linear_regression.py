import numpy as np
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def lin_predict(x, theta0, theta1):
    return theta0 + theta1 * x

def gradient_mse(x, y, theta0, theta1):
    # MSE = (1/n) * sum (r^2); r = y - yhat
    yhat = theta0 + theta1 * x
    r = y - yhat
    n = len(x)
    grad0 = (-2.0 / n) * np.sum(r)
    grad1 = (-2.0 / n) * np.sum(x * r)
    return grad0, grad1

def run_gd(x, y, lr=0.05, iters=1000):
    t0, t1 = 0.0, 0.0
    losses = []
    for i in range(iters):
        yhat = lin_predict(x, t0, t1)
        losses.append(mse(y, yhat))
        g0, g1 = gradient_mse(x, y, t0, t1)
        t0 -= lr * g0
        t1 -= lr * g1
    return t0, t1, np.array(losses)

def main(seed=42):
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.uniform(0, 5, size=n)
    noise = rng.normal(0, 1.0, size=n)
    y = 3.0 + 4.0 * x + noise

    # Closed-form
    X = np.column_stack([np.ones_like(x), x])
    theta_closed = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    t0_closed, t1_closed = theta_closed.tolist()

    # Gradient Descent
    t0_gd, t1_gd, losses = run_gd(x, y, lr=0.05, iters=1000)

    # Plot fitted lines
    plt.figure()
    plt.scatter(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    plt.plot(x_line, t0_closed + t1_closed * x_line, label="Closed-form fit")
    plt.plot(x_line, t0_gd + t1_gd * x_line, label="GD fit")
    plt.title("Synthetic Data and Fitted Lines")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.savefig("task7_fits.png", bbox_inches="tight")
    plt.close()

    # Plot loss curve
    plt.figure()
    plt.plot(np.arange(len(losses)), losses)
    plt.title("Gradient Descent Loss Curve (MSE vs Iterations)")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.savefig("task7_loss.png", bbox_inches="tight")
    plt.close()

    print("Closed-form:", t0_closed, t1_closed)
    print("Gradient Descent:", t0_gd, t1_gd)

if __name__ == "__main__":
    main()
