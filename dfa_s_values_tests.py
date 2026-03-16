# ====================== Tests for s_values in dfa function ======================


def test_dfa_s_values_truncate():
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    # 500 grater than 1000/4, must truncate input s_values to valid
    custom_s = [16, 32, 64, 128, 500]
    ref_s = [16, 32, 64, 128]
    # must warn DFA warning: only following S values are in use:
    with pytest.warns(UserWarning, match='only following S values are in use'):
        s_vals, f2_vals = dfa(dataset=data_2d, degree=2, s_values=custom_s)

    assert len(s_vals) == len(ref_s)  # Some must be filtered out
    assert all(s in ref_s for s in s_vals)


def test_dfa_s_values_all_invalid():
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    # must raise ValueError("... No valid scales found for analysis")
    invalid_s = [500, 600, 700]
    with pytest.raises(ValueError, match="No valid scales found for analysis"):
        dfa(dataset=data_2d, degree=2, s_values=invalid_s)


def test_dfa_single_s_value():
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    
    custom_s = 32
    
    s_vals, f2_vals = dfa(dataset=data_2d, degree=2, s_values=custom_s)
    assert len(s_vals)==1
    assert s_vals[0]==custom_s

    invalid_s = 500
    with pytest.raises(ValueError, match='Cannot use S > L / 4'):
        dfa(dataset=data_2d, degree=2, s_values=invalid_s)
    

def test_dfa_s_values_empty():
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    # must raise ValueError("Input s_values is empty.")
    custom_s = []
    with pytest.raises(ValueError, match="Input s_values is empty."):
        dfa(dataset=data_2d, degree=2, s_values=custom_s)


def test_dfa_s_values_empty():
    np.random.seed(42)
    data_2d = generate_fbn(hurst=0.5, length=1000, method="kasdin")
    # must raise ValueError("Input s_values is empty.")
    custom_s = []
    with pytest.raises(ValueError, match="Input s_values is empty."):
        dfa(dataset=data_2d, degree=2, s_values=custom_s)
