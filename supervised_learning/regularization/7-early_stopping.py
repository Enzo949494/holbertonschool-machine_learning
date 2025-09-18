def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if early stopping should occur.

    Args:
        cost: current validation cost
        opt_cost: lowest recorded validation cost
        threshold: threshold for improvement
        patience: patience count
        count: current count

    Returns:
        (should_stop, updated_count)
    """
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1
    return (count >= patience, count)