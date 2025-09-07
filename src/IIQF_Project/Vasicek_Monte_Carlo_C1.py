import numpy as np
import matplotlib.pyplot as plt

def vasicek_zcb_analytical(a, b, sigma, r0, T):
    """Analytical ZCB price under Vasicek (face=1)"""
    B = (1 - np.exp(-a * T)) / a
    A = np.exp((B - T)*(a*b - sigma**2/(2*a**2)) - (sigma**2 * B**2)/(4*a))
    P = A * np.exp(-B * r0)
    return P

def price_zcb_vasicek_mc_cv(
    a=0.0099, b=0.050269, sigma=0.015, r0=0.04,
    T=5.0, dt=1/252, n_paths=20000, n_plot=20, use_antithetic=True, random_seed=42
):
    np.random.seed(random_seed)

    n_steps = int(T / dt)
    exp_adt = np.exp(-a * dt)
    m_coef = b * (1 - exp_adt)
    var_coef = (sigma**2) * (1 - np.exp(-2*a*dt)) / (2*a)
    sqrt_var = np.sqrt(var_coef)

    # Generate standard normals
    half = n_paths // 2 if use_antithetic else n_paths
    Z = np.random.normal(size=(half, n_steps))
    if use_antithetic:
        Z = np.vstack([Z, -Z])

    # Simulate short-rate paths
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    for t in range(n_steps):
        r[:, t+1] = r[:, t] * exp_adt + m_coef + sqrt_var * Z[:, t]

    # Monte Carlo discounted payoffs
    integrals = (np.sum(r[:, :-1] + r[:, 1:], axis=1) * 0.5) * dt
    discounts = np.exp(-integrals)  # discounted payoff

    # --- Control Variates ---
    # Use the mean of simulated r_t over each path to compute a simpler proxy ZCB
    # Control: P_simple = exp(-r0*T) (ignoring stochasticity)
    P_control = np.exp(-r[:, 0]*T)  # simple discount using initial rate
    cov = np.cov(discounts, P_control)[0,1]
    var_control = np.var(P_control)
    beta = cov / var_control
    analytical_price = vasicek_zcb_analytical(a, b, sigma, r0, T)

    # Adjusted MC estimator
    discounts_cv = discounts - beta * (P_control - analytical_price)
    price_cv = discounts_cv.mean()
    std_error_cv = discounts_cv.std(ddof=1) / np.sqrt(n_paths)
    ci_low = price_cv - 1.96 * std_error_cv
    ci_high = price_cv + 1.96 * std_error_cv

    print("Monte Carlo Vasicek ZCB Pricing with Control Variates")
    print("------------------------------------------------------")
    print(f"Estimated price (CV): {price_cv:.6f}")
    print(f"Standard error (CV): {std_error_cv:.6f}")
    print(f"95% CI (CV): [{ci_low:.6f}, {ci_high:.6f}]")

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

    # Histogram of discounted payoffs
    plt.figure(figsize=(10,5))
    plt.hist(discounts_cv, bins=80, alpha=0.8)
    plt.title("Histogram of Discounted Payoffs (ZCB, CV adjusted, face=1)")
    plt.xlabel("Discounted payoff")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return {"price_cv": price_cv, "std_error_cv": std_error_cv, "ci_cv": (ci_low, ci_high), "discounts_cv": discounts_cv, "rates": r}

# Example run
res_cv = price_zcb_vasicek_mc_cv(
    a=0.0099, b=0.050269, sigma=0.015, r0=0.04,
    T=5.0, dt=1/252, n_paths=20000, n_plot=20, use_antithetic=True, random_seed=42
)
