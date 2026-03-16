def dfa(
    dataset,
    s_values: Union[int, Iterable, None] = None,
    degree: int = 2,
    processes: int = 1,
    n_integral: int = 1,
    s_min: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implementation of the Detrended Fluctuation Analysis (DFA) method.

    The algorithm removes local polynomial trends in integrated time series and
    returns the fluctuation function F^2(s) for each series.

    Args:
        dataset (ndarray): 1D or 2D array of time series data.
        s_values (Union[int, Iterable, None]): points where  fluctuation function F^2(s) is calculated (default: None).
        degree (int): Polynomial degree for detrending (default: 2).
        processes (int): Number of parallel workers (default: 1).
        n_integral (int): Number of cumulative sum operations to apply (default: 1).
        s_min (int): Minimal s_values (default: 5).

    Returns:
        tuple: (s, F2_s)
            - For 1D input: two 1D arrays s, F2_s.
            - For 2D input:
                s is a 1D array (same scales for all series),
                F2_s is a 2D array where each row is F^2(s) for one time series.
    """
    data = np.asarray(dataset, dtype=float)

    if data.ndim == 1:
        data = data.reshape(1, -1)
        single_series = True
    elif data.ndim == 2:
        single_series = False
    else:
        raise ValueError("Only 1D or 2D arrays are allowed!")

    series_len = data.shape[1]
    if s_values is None:
        s_max = int(series_len / 4)
        # if s_max < s_min:
        #     raise ValueError(
        #         f"Cannot create scales with s_max {s_max}<s_min {s_min}"
        #     )
        s_values = [int(exp(step)) for step in np.arange(np.log(s_min), np.log(s_max), 0.5)]
    elif isinstance(s_values, Iterable):
        init_s_len = len(s_values)
        if init_s_len < 1:
            raise ValueError("Input s_values is empty.")
        s_values = list(filter(lambda x: x <= series_len / 4, s_values))
        if len(s_values) < 1:
            raise ValueError(
                "Invalid s_values: all entries are greater than L / 4. "
                "No valid scales found for analysis."
            )

        if len(s_values) != init_s_len:
            warnings.warn(
                f'DFA warning: only following S values are in use: {s_values}'
                f'\nOriginal input had {init_s_len} values.',
                UserWarning,
                stacklevel=2)

    elif isinstance(s_values, (float, int)):
        if s_values > series_len / 4:
            raise ValueError("Cannot use S > L / 4")
        s_values = (s_values,)

    n_series = data.shape[0]
    results = None

    if processes <= 1:
        indices = np.arange(n_series)
        results = dfa_worker(
            indices=indices,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
        )
    else:
        processes = min(processes, cpu_count(), n_series)
        chunks = np.array_split(np.arange(n_series), processes)

        worker_func = partial(
            dfa_worker,
            arr=data,
            degree=degree,
            s_values=s_values,
            n_integral=n_integral,
        )

        results_list_of_lists = []
        with closing(Pool(processes=processes)) as pool:
            for sub in pool.map(worker_func, chunks):
                results_list_of_lists.append(sub)

        flat_results = []
        for sub in results_list_of_lists:
            flat_results.extend(sub)
        results = flat_results

    s_list = [r[0] for r in results]
    f2_list = [r[1] for r in results]

    if single_series:
        s_out = s_list[0]
        f2_out = f2_list[0]
    else:
        s_out = s_list[0]
        f2_out = np.vstack(f2_list)

    return s_out, f2_out
