from abtk.multiple_testing import benjamini_hochberg, bonferroni, holm_bonferroni


def test_bonferroni_basic():
    p = [0.01, 0.02, 0.5]
    adj = bonferroni(p)
    assert adj == [0.03, 0.06, 1.0]


def test_holm_bonferroni_monotonicity_and_order():
    p = [0.02, 0.01, 0.5]
    adj = holm_bonferroni(p)

    # Adjusted values should be within [0,1]
    assert all(0.0 <= x <= 1.0 for x in adj)

    # The smallest raw p-value should not end up with a larger adjustment than the second smallest
    # when comparing in sorted order.
    sorted_pairs = sorted(zip(p, adj), key=lambda t: t[0])
    assert sorted_pairs[0][1] <= sorted_pairs[1][1]


def test_bh_basic_properties():
    p = [0.01, 0.02, 0.04, 0.2]
    adj = benjamini_hochberg(p)

    assert all(0.0 <= x <= 1.0 for x in adj)

    # BH should be non-decreasing when raw p-values are sorted.
    sorted_pairs = sorted(zip(p, adj), key=lambda t: t[0])
    for i in range(1, len(sorted_pairs)):
        assert sorted_pairs[i][1] >= sorted_pairs[i - 1][1]


def test_bh_known_small_example():
    # A simple check against a hand-computed case.
    p = [0.01, 0.02, 0.5]
    # Sorted p: [0.01, 0.02, 0.5]
    # raw adjust: [0.03, 0.03, 0.5]
    # monotone from end => [0.03, 0.03, 0.5]
    adj = benjamini_hochberg(p)
    assert adj == [0.03, 0.03, 0.5]
