from __future__ import annotations

import time

import numpy as np

from lottogogo.engine.sampler import MonteCarloSampler


def _uniform_probabilities() -> dict[int, float]:
    return {number: 1.0 for number in range(1, 46)}


def test_t311_weighted_sampling_prefers_higher_probability_numbers():
    probabilities = {number: 0.6 / 44 for number in range(1, 45)}
    probabilities[45] = 0.4

    sampler = MonteCarloSampler(sample_size=4000, chunk_size=2000)
    samples = sampler.sample(probabilities, seed=123)

    count_45 = sum(1 for combo in samples if 45 in combo)
    count_1 = sum(1 for combo in samples if 1 in combo)

    assert all(len(combo) == 6 for combo in samples)
    assert count_45 > count_1 * 3


def test_t312_no_duplicates_inside_each_combination():
    sampler = MonteCarloSampler(sample_size=3000, chunk_size=1500)
    samples = sampler.sample(_uniform_probabilities(), seed=42)

    for combo in samples:
        assert len(combo) == 6
        assert len(set(combo)) == 6
        assert tuple(sorted(combo)) == combo


def test_t313_large_generation_performance_and_memory():
    sampler = MonteCarloSampler(sample_size=100000, chunk_size=20000)
    start = time.perf_counter()
    sampled = sampler.sample_array(_uniform_probabilities(), seed=7)
    elapsed = time.perf_counter() - start

    assert sampled.shape == (100000, 6)
    assert elapsed < 5.0
    assert sampled.nbytes < 500 * 1024 * 1024
    assert np.all(np.diff(sampled, axis=1) > 0)

