from abtk.power import (
    power_two_means,
    power_two_proportions,
    sample_size_two_means,
    sample_size_two_proportions,
)


def test_sample_size_two_proportions_reasonable_direction():
    n = sample_size_two_proportions(0.08, 0.01, alpha=0.05, power=0.8)
    assert n > 0

    # Harder problem (smaller effect) should require more users.
    n2 = sample_size_two_proportions(0.08, 0.005, alpha=0.05, power=0.8)
    assert n2 > n


def test_power_two_proportions_increases_with_n():
    p_small = power_two_proportions(0.08, 0.01, n_per_group=2000, alpha=0.05)
    p_large = power_two_proportions(0.08, 0.01, n_per_group=20000, alpha=0.05)
    assert p_large > p_small


def test_sample_size_two_means_reasonable_direction():
    n = sample_size_two_means(standard_deviation=10.0, minimum_detectable_effect=1.0, power=0.8)
    assert n > 0

    # Smaller effect should require more users.
    n2 = sample_size_two_means(standard_deviation=10.0, minimum_detectable_effect=0.5, power=0.8)
    assert n2 > n


def test_power_two_means_increases_with_n():
    p_small = power_two_means(
        standard_deviation=10.0, minimum_detectable_effect=1.0, n_per_group=200, alpha=0.05
    )
    p_large = power_two_means(
        standard_deviation=10.0, minimum_detectable_effect=1.0, n_per_group=2000, alpha=0.05
    )
    assert p_large > p_small
