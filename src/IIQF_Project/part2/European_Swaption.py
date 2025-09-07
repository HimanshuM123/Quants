import math
import numpy as np
import pandas as pd

def build_sofr_curve(deposit_rates, swap_par_rates, max_t=7.0, freq=2):
    """
    deposit_rates: dict mapping tenor_years -> continuous annual rate (decimal), e.g. {0.5:0.0408}
    swap_par_rates: dict mapping tenor_years -> par swap fixed rate (decimal), e.g. {1:0.04071,2:0.03719,...}
    max_t: max horizon to return (years)
    freq: number of fixed payments per year (2 for semi-annual)
    """
    alpha = 1.0 / freq
    # create semiannual nodes
    times = np.arange(0.0, max_t + 1e-12, alpha)
    # known discount factors dict
    DF = {0.0:1.0}
    # seed from deposit rates (short tenors)
    for tenor, r in deposit_rates.items():
        DF[round(tenor,6)] = math.exp(-r * tenor)

    # sort swap pillar tenors
    pillars = sorted(swap_par_rates.keys())

    # bootstrap for pillars where possible:
    # we will attempt pillars in ascending order and solve for DF(T) when all earlier fixed payment dates' DFs are known.
    for T in pillars:
        S = swap_par_rates[T]
        N = int(round(T * freq))
        # sum of known alpha * D(t_i) for i=1..N-1
        sum_known = 0.0
        unknown_indices = []
        for i in range(1, N):
            t_i = round(i*alpha,6)
            if t_i in DF:
                sum_known += alpha * DF[t_i]
            else:
                unknown_indices.append(t_i)
        # If no unknown intermediate DF, we can solve for DF(T) directly:
        if len(unknown_indices) == 0:
            # S = (1 - D_T) / (sum_known + alpha * D_T)  -> solve for D_T
            D_T = (1.0 - S * sum_known) / (1.0 + S * alpha)
            DF[round(T,6)] = D_T
        else:
            # If some intermediate DF missing, we cannot uniquely solve without additional instruments.
            # A pragmatic approach: assume intermediate zero rates follow linear interpolation between last known pillar and T,
            # solve for D_T using that assumption. (This is an approximation.)
            # Find last known time < smallest unknown
            known_times = sorted([t for t in DF.keys() if t>0])
            if not known_times:
                raise ValueError("Insufficient short-rate inputs to bootstrap.")
            t0 = known_times[-1]
            z0 = -math.log(DF[t0])/t0
            # make an initial guess for z_T = S (simple)
            zT_guess = S
            # linear interpolate zero rates for unknowns between t0 and T
            for t in unknown_indices + [round(T,6)]:
                frac = (t - t0) / (T - t0)
                zt = z0 + frac * (zT_guess - z0)
                DF[t] = math.exp(-zt * t)
            # now recompute sum_known (should be complete) and solve for D_T exactly
            sum_known = sum(alpha * DF[round(i*alpha,6)] for i in range(1, N))
            D_T = (1.0 - S * sum_known) / (1.0 + S * alpha)
            DF[round(T,6)] = D_T

    # compute zero rates at pillar times
    zero = {t: (-math.log(DF[t]) / t) if t>0 else 0.0 for t in DF.keys()}

    # interpolate zero rates linearly to every semiannual node
    times_sorted = np.sort(times)
    zero_interp = {}
    pillar_times = sorted([t for t in zero.keys() if t>0])
    for t in times_sorted:
        if t in zero:
            zero_interp[round(t,6)] = zero[t]
            continue
        # find surrounding pillar times
        left = max([p for p in pillar_times if p < t], default=None)
        right = min([p for p in pillar_times if p > t], default=None)
        if left is None:
            zero_interp[round(t,6)] = zero[right]
        elif right is None:
            zero_interp[round(t,6)] = zero[left]
        else:
            zt = zero[left] + (zero[right] - zero[left]) * ((t - left) / (right - left))
            zero_interp[round(t,6)] = zt

    # generate final DF table
    rows = []
    for t in times_sorted:
        z = zero_interp[round(t,6)]
        df = math.exp(-z * t)
        rows.append((round(t,6), df, z))
    df_out = pd.DataFrame(rows, columns=["T_years","DiscountFactor","ZeroRate_continuous"])
    df_out["ZeroRate_%"] = df_out["ZeroRate_continuous"] * 100.0
    return df_out

# Example usage with the sample quotes I cited earlier (replace with live quotes in production):
deposit_rates_example = {0.5: 0.0408}   # 6M continuous SOFR example
swap_par_rates_example = {1:0.04071, 2:0.03719, 3:0.03639, 5:0.036985, 7:0.038347}
curve = build_sofr_curve(deposit_rates_example, swap_par_rates_example, max_t=7.0, freq=2)
print(curve.to_string(index=False))
