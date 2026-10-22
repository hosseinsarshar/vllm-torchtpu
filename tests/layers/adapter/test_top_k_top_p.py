# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_torchtpu.envs as envs
from vllm_torchtpu.layers.adapter.sample import top_k_top_p as sampling
from vllm_torchtpu.runner.tpu_runner import TPUModelRunner


def test_int32_search_boundaries():
    thresholds = torch.tensor(
        [[-2147483648, -37, -1], [0, 128, 2147483647]], dtype=torch.int32
    )
    result = sampling._int32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    torch.testing.assert_close(result, thresholds)


def test_float32_search_boundaries():
    maximum = torch.finfo(torch.float32).max
    thresholds = torch.tensor([-maximum, -10.5, -1.0, 0.0, 3.25, maximum])
    result = sampling._float32_bsearch(
        thresholds.shape, lambda x: x > thresholds, thresholds.device
    )
    torch.testing.assert_close(
        result, thresholds, rtol=0, atol=torch.finfo(torch.float32).tiny
    )


@pytest.mark.parametrize("shape", [(5,), (2, 5), (2, 3, 5)])
@pytest.mark.parametrize("k", [1, 2, 5])
def test_topk_matches_sorted_cutoff(shape, k):
    logits = torch.tensor([-4.0, 3.0, 1.0, 3.0, -2.0]).expand(shape)
    cutoff = logits.sort(dim=-1).values[..., -k, None]
    expected = torch.where(logits >= cutoff, logits, -999.0)
    result = sampling.topk_mask(logits, torch.full((*shape[:-1], 1), k), -999.0)
    torch.testing.assert_close(result, expected)


@pytest.mark.parametrize(
    "p, keep",
    [
        (0.1, [True, False, False, False, False]),
        (0.65, [True, True, True, False, False]),
        (1.0, [True, True, True, True, True]),
    ],
)
def test_topp_includes_cutoff_ties(p, keep):
    logits = torch.log(torch.tensor([0.5, 0.2, 0.2, 0.08, 0.02]))
    expected = torch.where(torch.tensor(keep), logits, -999.0)
    result = sampling.topp_mask(logits, torch.tensor([p]), -999.0)
    torch.testing.assert_close(result, expected)


def test_apply_top_k_top_p_per_request():
    probabilities = np.array(
        [
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
            [0.40, 0.30, 0.20, 0.10],
        ],
        dtype=np.float32,
    )
    logits = torch.from_numpy(np.log(probabilities))
    top_k = torch.tensor([[0], [2], [-1], [3]])
    top_p = torch.tensor([[1.0], [1.0], [0.65], [0.5]])
    # In row 3, top-k renormalizes .4/.3/.2; top-p needs two tokens to reach .5.
    keep = torch.tensor(
        [
            [True, True, True, True],
            [True, True, False, False],
            [True, True, False, False],
            [True, True, False, False],
        ]
    )
    expected = torch.where(keep, logits, sampling.MASKED_LOGIT_VALUE)
    original = logits.clone()
    result = sampling.apply_top_k_top_p(logits, top_k, top_p)
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(logits, original)


def test_candidate_blocks_match_full_vocab_for_top_k_sampling():
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(6, 128 * 400, generator=gen) * 3).bfloat16().float() / 0.6
    top_k = torch.full((6, 1), 20, dtype=torch.int32)
    top_p = torch.full((6, 1), 0.95)
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, 64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_candidate_blocks_keep_ties_within_the_documented_bound():
    # 19 distinct large logits plus 13 ties at the 20th value, one per block:
    # 32 logits >= the 20th largest, exactly num_blocks.
    num_blocks, k = 32, 20
    logits = torch.full((1, 128 * 100), -5.0)
    for i in range(k - 1):
        logits[0, 128 * i] = 10.0 + i
    for j in range(num_blocks - k + 1):
        logits[0, 128 * (k - 1 + j) + 7] = 9.0
    top_k = torch.tensor([[k]], dtype=torch.int32)
    for p in (1.0, 0.9):
        top_p = torch.tensor([[p]])
        expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
        result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, num_blocks)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_candidate_blocks_leave_rows_without_top_k_unmasked():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 128 * 50, generator=gen)
    top_k = torch.tensor([[0], [-1], [5]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [1.0], [0.9]])
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_candidates(logits, top_k, top_p, 16)
    torch.testing.assert_close(result[:2], logits[:2], rtol=0, atol=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_topk_candidate_blocks_gating(monkeypatch):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vocab_size = 248320

    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPK_CANDIDATES", 0)
    assert runner._topk_candidate_blocks() == 0

    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPK_CANDIDATES", 64)
    runner.vocab_size = 248321
    assert runner._topk_candidate_blocks() == 0
    runner.vocab_size = 248320
    assert runner._topk_candidate_blocks() == 64


def test_sampler_topk_candidates_gating(monkeypatch):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vocab_size = 248320
    monkeypatch.setattr(envs, "TPU_SAMPLER_TOPK_CANDIDATES", 64)

    input_batch = SimpleNamespace(
        top_k_cpu_tensor=torch.tensor([20, 20, 20, 20], dtype=torch.int64)
    )
    runner.input_batch = input_batch

    assert runner._sampler_topk_candidates(0, 4, None) == 64
    assert runner._sampler_topk_candidates(0, 4, object()) == 0

    input_batch.top_k_cpu_tensor = torch.tensor([40, 20, 20, 20], dtype=torch.int64)
    assert runner._sampler_topk_candidates(0, 4, None) == 0

    input_batch.top_k_cpu_tensor = torch.tensor([0, 20, 20, 20], dtype=torch.int64)
    assert runner._sampler_topk_candidates(0, 4, None) == 0


def test_rejection_sampler_topk_candidates_cpu_parity():
    gen = torch.Generator().manual_seed(42)
    vocab_size = 128 * 400
    rows = 4
    logits = (torch.randn(rows, vocab_size, generator=gen) * 3).float() / 0.6
    top_k = torch.full((rows, 1), 20, dtype=torch.int32)
    top_p = torch.full((rows, 1), 0.95)

    expected = sampling.apply_top_k_top_p(logits, top_k, top_p, topk_candidates=0)
    result = sampling.apply_top_k_top_p(logits, top_k, top_p, topk_candidates=64)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "num_values, expected",
    [
        (151936, 64),  # Qwen3: 128 * 1187.
        (248320, 64),  # Qwen3.5.
        (202048, 56),  # Llama 4: not a multiple of 128 or 64.
        (128256, 48),  # Llama 3.
        (12628, 14),  # No multiple of 8 divides it: any divisor.
        (4099, 0),  # Prime: plain top-k.
        (256, 0),  # Too few values for a chunk to help.
    ],
)
def test_topk_prefilter_chunk(num_values, expected):
    assert sampling.topk_prefilter_chunk(num_values, 64) == expected


@pytest.mark.parametrize("num_values", [151936, 202048, 12628, 4099])
def test_prefiltered_top_k_matches_torch_topk(num_values):
    # Rows of distinct values, so the indices must match torch.topk too.
    gen = torch.Generator().manual_seed(num_values)
    values = torch.stack(
        [torch.randperm(num_values, generator=gen) for _ in range(3)]
    ).float()
    expected_values, expected_indices = torch.topk(values, 64)
    top_values, top_indices = sampling.prefiltered_top_k(values, 64)
    torch.testing.assert_close(top_values, expected_values, rtol=0, atol=0)
    torch.testing.assert_close(top_indices, expected_indices, rtol=0, atol=0)


@pytest.mark.parametrize("num_allowed", [5, 300, None])
def test_prefiltered_top_k_with_ties_and_masked_logits(num_allowed):
    # Few distinct values (many ties), and grammar-style -inf masks.
    gen = torch.Generator().manual_seed(7)
    values = torch.randint(0, 8, (4, 151936), generator=gen).float()
    if num_allowed is not None:
        allowed = torch.zeros_like(values, dtype=torch.bool)
        for row in allowed:
            row[torch.randperm(row.numel(), generator=gen)[:num_allowed]] = True
        values = torch.where(allowed, values, float("-inf"))
    expected_values, _ = torch.topk(values, 64)
    top_values, top_indices = sampling.prefiltered_top_k(values, 64)
    torch.testing.assert_close(top_values, expected_values, rtol=0, atol=0)
    # Ties may pick other indices, but each index points at its value once.
    torch.testing.assert_close(
        torch.gather(values, 1, top_indices), top_values, rtol=0, atol=0
    )
    for row in top_indices:
        assert row.unique().numel() == row.numel()


@pytest.mark.parametrize("vocab_size", [151936, 202048])
def test_prefilter_matches_full_vocab_for_top_k_sampling(vocab_size):
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(6, vocab_size, generator=gen) * 3).bfloat16().float() / 0.6
    top_k = torch.tensor([[20], [1], [32], [5], [20], [20]], dtype=torch.int32)
    top_p = torch.tensor([[0.95], [1.0], [0.9], [0.5], [1.0], [0.8]])
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p(logits, top_k, top_p, topk_prefilter=True)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_prefilter_keeps_ties_within_the_documented_bound():
    # 19 distinct large logits plus 45 ties at the 20th value: 64 logits
    # >= the 20th largest, exactly TOPK_PREFILTER_CANDIDATES.
    num_candidates, k = sampling.TOPK_PREFILTER_CANDIDATES, 20
    logits = torch.full((1, 151936), -5.0)
    for i in range(k - 1):
        logits[0, 997 * i] = 10.0 + i
    for j in range(num_candidates - k + 1):
        logits[0, 997 * (k - 1 + j) + 7] = 9.0
    top_k = torch.tensor([[k]], dtype=torch.int32)
    for p in (1.0, 0.9):
        top_p = torch.tensor([[p]])
        expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
        result = sampling.apply_top_k_top_p_prefilter(logits, top_k, top_p)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_prefilter_leaves_rows_without_top_k_unmasked():
    gen = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 151936, generator=gen)
    top_k = torch.tensor([[0], [-1], [5]], dtype=torch.int32)
    top_p = torch.tensor([[1.0], [1.0], [0.9]])
    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p_prefilter(logits, top_k, top_p)
    torch.testing.assert_close(result[:2], logits[:2], rtol=0, atol=0)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_prefilter_sampling_matches_full_vocab_gumbel_max():
    # Given the full-vocab noise at each candidate's token, the compact
    # sampler picks the same token as masking the full row and taking the
    # Gumbel-max over it.
    gen = torch.Generator().manual_seed(3)
    rows, vocab_size = 8, 151936
    scaled = (torch.randn(rows, vocab_size, generator=gen) * 3).bfloat16().float()
    top_k = torch.tensor([[20], [5], [32], [1], [20], [10], [20], [32]])
    top_p = torch.tensor([[0.95], [1.0], [0.8], [1.0], [0.5], [0.99], [1.0], [0.9]])
    u_full = torch.rand(rows, vocab_size, generator=gen)
    finfo = torch.finfo(torch.float32)
    gumbel = -torch.log(-torch.log(u_full.clamp(min=finfo.tiny, max=1.0 - finfo.eps)))
    masked = sampling.apply_top_k_top_p(scaled, top_k, top_p)
    expected = torch.argmax(masked + gumbel, dim=-1, keepdim=True)

    _, top_indices = sampling.prefiltered_top_k(
        scaled, sampling.TOPK_PREFILTER_CANDIDATES
    )
    u = torch.gather(u_full, 1, top_indices)
    result = sampling.sample_top_k_top_p_prefilter(scaled, top_k, top_p, u)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_sampler_topk_prefilter_gating(monkeypatch):
    runner = TPUModelRunner.__new__(TPUModelRunner)
    runner.vocab_size = 151936
    input_batch = SimpleNamespace(
        top_k_cpu_tensor=torch.tensor([20, 20, 20, 20], dtype=torch.int64)
    )
    runner.input_batch = input_batch

    monkeypatch.setattr(envs, "ENABLE_TOPK_PREFILTER", False)
    assert not runner._topk_prefilter_enabled()
    assert not runner._sampler_topk_prefilter(0, 4, None)

    monkeypatch.setattr(envs, "ENABLE_TOPK_PREFILTER", True)
    assert runner._topk_prefilter_enabled()
    assert runner._sampler_topk_prefilter(0, 4, None)
    assert not runner._sampler_topk_prefilter(0, 4, object())

    input_batch.top_k_cpu_tensor = torch.tensor([33, 20, 20, 20], dtype=torch.int64)
    assert not runner._sampler_topk_prefilter(0, 4, None)
    assert runner._sampler_topk_prefilter(1, 4, None)

    input_batch.top_k_cpu_tensor = torch.tensor([0, 20, 20, 20], dtype=torch.int64)
    assert not runner._sampler_topk_prefilter(0, 4, None)

    # Unlike TPU_SAMPLER_TOPK_CANDIDATES, any vocab size works.
    input_batch.top_k_cpu_tensor = torch.tensor([32, 1, 20, 20], dtype=torch.int64)
    runner.vocab_size = 202048
    assert runner._sampler_topk_prefilter(0, 4, None)


def test_rejection_sampler_topk_prefilter_cpu_parity():
    gen = torch.Generator().manual_seed(42)
    vocab_size = 151936
    rows = 4
    logits = (torch.randn(rows, vocab_size, generator=gen) * 3).float() / 0.6
    top_k = torch.full((rows, 1), 20, dtype=torch.int32)
    top_p = torch.full((rows, 1), 0.95)

    expected = sampling.apply_top_k_top_p(logits, top_k, top_p)
    result = sampling.apply_top_k_top_p(
        logits, top_k, top_p, topk_candidates=64, topk_prefilter=True
    )
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
