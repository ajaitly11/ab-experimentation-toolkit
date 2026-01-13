from abtk.health import srm_check


def test_srm_check_no_issue_for_balanced_split():
    # 50/50 split, counts match exactly
    res = srm_check(50000, 50000, expected_split=(0.5, 0.5))
    assert res.p_value > 0.05


def test_srm_check_flags_large_mismatch():
    # Intended 50/50 but observed 60/40: large enough to be a serious issue at scale
    res = srm_check(60000, 40000, expected_split=(0.5, 0.5))
    assert res.p_value < 1e-6


def test_srm_check_supports_custom_split():
    # Intended 90/10 and observed close to it
    res = srm_check(9000, 1000, expected_split=(0.9, 0.1))
    assert res.p_value > 0.05
