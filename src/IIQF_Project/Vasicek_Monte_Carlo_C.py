import numpy as np
import matplotlib.pyplot as plt

def price_zcb_vasicek_mc_plot_paths(
    a=0.0099, b=0.050269, sigma=0.015, r0=0.04,
    T=5.0, dt=1/252, n_paths=20000, n_plot=20, use_antithetic=True, random_seed=42
):
    np.random.seed(random_seed)

    n_steps = int(T / dt)
    exp_adt = np.exp(-a * dt)
    m_coef = b * (1 - exp_adt)   # deterministic shift each step
    var_coef = (sigma**2) * (1 - np.exp(-2*a*dt)) / (2*a)
    sqrt_var = np.sqrt(var_coef)

    # generate standard normals (half if antithetic)
    half = n_paths // 2 if use_antithetic else n_paths
    Z = np.random.normal(size=(half, n_steps))
    if use_antithetic:
        Z = np.vstack([Z, -Z])

    # allocate and set initial rates
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0

    # simulate using exact OU discretization
    for t in range(n_steps):
        r[:, t+1] = r[:, t] * exp_adt + m_coef + sqrt_var * Z[:, t]

    # integrate r over time using trapezoid rule -> integrals shape (n_paths,)
    integrals = (np.sum(r[:, :-1] + r[:, 1:], axis=1) * 0.5) * dt
    discounts = np.exp(-integrals)  # discounted payoff for face=1

    price = discounts.mean()
    std_error = discounts.std(ddof=1) / np.sqrt(n_paths)
    ci_low = price - 1.96 * std_error
    ci_high = price + 1.96 * std_error

    print("Monte Carlo Vasicek ZCB Pricing")
    print("--------------------------------")
    print(f"Parameters: a={a}, b={b}, sigma={sigma}, r0={r0}, T={T}, dt={dt}")
    print(f"Paths: {n_paths}, antithetic={use_antithetic}")
    print(f"Estimated price: {price:.6f}")
    print(f"Standard error: {std_error:.6f}")
    print(f"95% CI: [{ci_low:.6f}, {ci_high:.6f}]")

    # Plot some short-rate paths
    plt.figure(figsize=(12,6))
    time_grid = np.linspace(0, T, n_steps + 1)
    for i in range(min(n_plot, n_paths)):
        plt.plot(time_grid, r[i, :], lw=1, alpha=0.7)
    plt.title(f"Monte Carlo Vasicek Short-Rate Paths (first {n_plot} paths)")
    plt.xlabel("Time (years)")
    plt.ylabel("Short Rate")
    plt.grid(True)
    plt.show()

    # histogram of discounted payoffs
    plt.figure(figsize=(10,5))
    plt.hist(discounts, bins=80, alpha=0.8)
    plt.title("Histogram of Discounted Payoffs (ZCB, face=1)")
    plt.xlabel("Discounted payoff")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return {"price": price, "std_error": std_error, "ci": (ci_low, ci_high), "discounts": discounts, "rates": r}

# Example run
res = price_zcb_vasicek_mc_plot_paths(
    a=0.0099, b=0.050269, sigma=0.015, r0=0.04,
    T=5.0, dt=1/252, n_paths=20000, n_plot=20, use_antithetic=True, random_seed=42
)
