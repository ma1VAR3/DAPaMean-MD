import math
import os
import numpy as np


def get_left_right_counts(bins, values):
    left_counts = []
    right_counts = []

    for v in bins:
        left_counts.append(len([x for x in values if x < v]))
        right_counts.append(len([x for x in values if x > v]))

    return left_counts, right_counts


def get_probs(bins, left_counts, right_counts, epsilon, factor):
    c = np.maximum(left_counts, right_counts)
    probs = [math.exp(-(epsilon * c[i]) / (2 * factor)) for i in range(len(bins))]
    probs = probs / np.sum(probs)
    # print("Probability assigned to quantized means: ", probs)
    return probs


def get_probs_quantiles(vals, alpha, epsilon, factor):
    k = len(vals) - 1
    probs = [
        (vals[i + 1] - vals[i])
        * (math.exp(-(epsilon / (2 * factor)) * abs(i - (alpha * k))))
        for i in range(len(vals) - 1)
    ]
    probs = probs / np.sum(probs)

    return probs


def private_quantile(vals, q, epsilon, ub, lb, num_vals, factor):
    vals_c = [lb if v < lb else ub if v > ub else v for v in vals]
    vals_sorted = np.sort(vals_c)
    new_s_vals = [lb]
    new_s_vals = np.append(new_s_vals, vals_sorted)
    new_s_vals = np.append(new_s_vals, ub)
    probs = get_probs_quantiles(new_s_vals, q, epsilon, factor)
    indices = np.arange(0, len(new_s_vals) - 1)
    selected_interval = np.random.choice(indices, num_vals, p=probs)
    selected_quantile = [
        np.random.uniform(
            new_s_vals[selected_interval[i]], new_s_vals[selected_interval[i] + 1]
        )
        for i in range(len(selected_interval))
    ]
    return selected_quantile


def private_estimation(
    user_group_means,
    L,
    K,
    ub,
    lb,
    epsilon,
    num_exp,
    actual_mean,
    groupping_algo,
    conc_algo,
    config,
):
    file_base = (
        "./results/"
        + conc_algo
        + "/"
        + groupping_algo
        + "/"
        + "epsilon_"
        + str(epsilon)
        + "/"
    )

    if conc_algo == "coarse_mean":
        tau = config["tau"]
        delta = config["delta"]
        # for tau in taus:
        # Quantizing means
        if tau == -1:
            tau = ((ub - lb) / 2) * (math.sqrt((2 / L) * (math.log((2 * K) / delta))))
            # print("Levy tau: ", tau)
        quantized_bins = np.arange(lb + tau / 2, ub, tau)
        factor = 2 if groupping_algo == "wrap" else 1
        diff_matrix = np.subtract.outer(user_group_means, quantized_bins)
        idx = np.abs(diff_matrix).argmin(axis=1)
        quantized_means = quantized_bins[idx]
        quantized_means = np.sort(quantized_means)

        # Assigning probabilities to quantized means
        left_counts, right_counts = get_left_right_counts(
            quantized_bins, quantized_means
        )
        probs = get_probs(
            quantized_bins, left_counts, right_counts, epsilon / 2, factor
        )

        # Selecting quantized means and projecting them
        selected_quantized_means = np.random.choice(quantized_bins, num_exp, p=probs)
        ub_calc = selected_quantized_means + 1.5 * tau
        lb_calc = selected_quantized_means - 1.5 * tau
        lb_calc = [
            lb if l < lb else (ub - (3 * tau)) if l > (ub - (3 * tau)) else l
            for l in lb_calc
        ]
        ub_calc = [
            (lb + (3 * tau)) if u < (lb + (3 * tau)) else ub if u > ub else u
            for u in ub_calc
        ]
        projected_vals = [
            np.clip(user_group_means, lb_calc[i], ub_calc[i])
            for i in range(len(lb_calc))
        ]
        mean_of_projected_vals = np.mean(projected_vals, axis=1)
        noise_projected_vals = np.random.laplace(
            0, (3 * tau * factor) / (K * (epsilon / 2)), num_exp
        )
        # print("Noise scale: ", (3 * tau * factor) / (K * (epsilon / 2)))
        final_estimates = mean_of_projected_vals + noise_projected_vals
        final_estimates = np.clip(final_estimates, lb, ub)
        losses = np.abs(final_estimates - actual_mean)

        return losses

    elif conc_algo == "quantiles":
        q = config["lower_quantile"]
        # for q in quantiles:

        user_group_means = np.sort(user_group_means)
        factor = 2 if groupping_algo == "wrap" else 1
        quantile_1 = q
        quantile_2 = 1 - q
        q1_t = private_quantile(
            user_group_means, quantile_1, epsilon / 4, ub, lb, num_exp, factor
        )
        q2_t = private_quantile(
            user_group_means, quantile_2, epsilon / 4, ub, lb, num_exp, factor
        )
        q1 = np.minimum(q1_t, q2_t)
        q2 = np.maximum(q1_t, q2_t)
        projected_vals = [
            np.clip(user_group_means, q1[i], q2[i]) for i in range(len(q1))
        ]

        mean_of_projected_vals = np.mean(projected_vals, axis=1)
        noise_projected_vals = [
            np.random.laplace(0, (((q2[i] - q1[i]) * factor) / (K * (epsilon / 2))))
            for i in range(len(q1))
        ]
        final_estimates = mean_of_projected_vals + noise_projected_vals
        final_estimates = np.clip(final_estimates, lb, ub)
        losses = np.abs(final_estimates - actual_mean)
        # np.save(file_base_q + 'losses.npy', losses)
        # statistical_losses = np.abs(mean_of_projected_vals - actual_mean)
        # np.save(file_base_q + 'statistical_losses.npy', statistical_losses)
        # random_losses = np.abs(noise_projected_vals)
        # np.save(file_base_q + 'random_losses.npy', random_losses)

        return losses

    if conc_algo == "baseline2":
        factor = 2 if groupping_algo == "wrap" else 1
        coarse_mean = np.mean(user_group_means)
        noise_baseline2 = np.random.laplace(0, ((ub - lb) / (K * epsilon)), num_exp)
        final_estimates = coarse_mean + noise_baseline2
        final_estimates = np.clip(final_estimates, lb, ub)
        losses = np.abs(final_estimates - actual_mean)
        return losses


def baseline_estimation(data, ub, lb, e, actual_mean, num_exp):
    data_grouped = data.groupby(["User"]).agg({"Value": "count"}).reset_index()
    max_contrib = np.max(data_grouped["Value"])
    sum_contrib = np.sum(data_grouped["Value"])
    # print("Max contribution: ", max_contrib)
    # print("Sum of contributions: ", sum_contrib)

    b = ((ub - lb) * max_contrib) / (sum_contrib * e)
    noise = np.random.laplace(0, b, num_exp)
    final_estimates = actual_mean + noise
    # final_estimates = [ub if f > ub else lb if f < lb else f for f in final_estimates]
    final_estimates = np.clip(final_estimates, lb, ub)
    losses = np.abs(final_estimates - actual_mean)

    return losses
