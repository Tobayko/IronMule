"""The single, runtime-independent contract for H0 correctness fixtures.

The arrays themselves are deliberately not generated here: the benchmark is
the only producer and owns its lazy NumPy import.  This module binds the
producer's deterministic PCG64/FP32-to-little-endian-FP16 algorithm to the
metadata and digests recorded at the contract boundary.  If NumPy or the
generator changes, these digests intentionally fail closed until the contract
is consciously re-registered.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from typing import Any


class CorrectnessContractError(ValueError):
    """Raised when a correctness case is not the registered fixture."""


CORRECTNESS_CASES = (
    ("visible_64", (64, 64), (64, 64), 0xC0DE0001, -1.0, 1.0, False),
    ("visible_rectangular", (17, 31), (31, 13), 0xC0DE0002, -1.0, 1.0, False),
    ("visible_zero_rhs", (33, 65), (65, 7), 0xC0DE0003, -1.0, 1.0, True),
    ("visible_small_range", (31, 47), (47, 19), 0xC0DE0004, -(2**-10), 2**-10, False),
    ("visible_large_range", (31, 47), (47, 19), 0xC0DE0005, -4.0, 4.0, False),
    ("holdout_uniform", (23, 37), (37, 29), 0xC0DE1001, -1.0, 1.0, False),
    ("holdout_large_range", (65, 33), (33, 9), 0xC0DE1002, -4.0, 4.0, False),
)
CORRECTNESS_CASE_NAMES = tuple(case[0] for case in CORRECTNESS_CASES)
CORRECTNESS_CASE_SPECS = {
    case[0]: {
        "shape": [*case[1], *case[2]],
        "seed": case[3],
        "low": case[4],
        "high": case[5],
        "zero_rhs": case[6],
    }
    for case in CORRECTNESS_CASES
}
PERFORMANCE_CASE_NAME = "performance_fixture"
SIGN_INVARIANT_CASE_NAME = "sign_invariant"
ALL_CORRECTNESS_CASE_NAMES = (*CORRECTNESS_CASE_NAMES, PERFORMANCE_CASE_NAME, SIGN_INVARIANT_CASE_NAME)
CORRECTNESS_HARD_CAPS = {
    "abs_max": 1.0,
    "rel_q99_abs_oracle_ge_1": 0.05,
    "normalized_l2": 0.01,
}
CORRECTNESS_FULL_KEYS = frozenset(
    {
        "name", "shape", "dtype", "layout", "seed", "a_sha256", "b_sha256",
        "fixture_digest", "zero_rhs", "metrics", "passed", "hard_caps",
    }
)
# Numeric sign-invariant metrics are not independently reconstructible by the
# offline worker/aggregator.  The producer therefore reports only the closed
# relation and the trusted fixture binding; the actual comparison remains a
# producer-side correctness gate.
SIGN_INVARIANT_KEYS = frozenset({"name", "seed", "fixture_digest", "reference", "relation", "passed"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

# These values are intentionally shared by the producer, parent normalizer,
# and pure aggregator.  A missing value is valid only with one of these closed
# reasons; arbitrary source strings would make ``not_evaluable`` ambiguous.
MEMORY_MISSING_REASONS = frozenset(
    {
        "not_recorded", "not_applicable", "api_unavailable", "unavailable", "no_sample",
        "source_missing", "no_oracle_elements_abs_ge_1", "invalid_source_value",
        "invalid_api_payload", "entry_limit", "ps_exit", "ps_parse", "ps_negative",
        "parent_setup_failure",
    }
)
MEMORY_API_ERROR_TYPES = frozenset(
    {
        "AttributeError", "ImportError", "MemoryError", "OSError", "OverflowError",
        "RuntimeError", "TypeError", "ValueError", "other",
    }
)
MEMORY_MAX_INT = (1 << 63) - 1
MEMORY_LIMIT_KEYS = frozenset({"attempted", "hard_limit", "applied", "missing_reason"})
MEMORY_METRIC_NAMES = frozenset(
    {"mlx_active_memory", "mlx_peak_memory", "mlx_cache_memory", "rss", "custom", "memory_metrics"}
)


def memory_name_allowed(name: Any) -> bool:
    """Return whether a memory telemetry name is part of the shared schema."""

    return isinstance(name, str) and name in MEMORY_METRIC_NAMES


def memory_api_error_reason(exc: BaseException) -> str:
    """Map an API exception to a registered, non-forgeable reason."""

    error_type = type(exc).__name__
    if error_type not in MEMORY_API_ERROR_TYPES:
        error_type = "other"
    return f"api_error:{error_type}"


def missing_reason_allowed(reason: Any) -> bool:
    """Return whether a missing telemetry/correctness value has a closed reason."""

    if not isinstance(reason, str) or not reason or len(reason) > 256 or "\x00" in reason:
        return False
    if reason in MEMORY_MISSING_REASONS:
        return True
    return reason.startswith("api_error:") and reason.removeprefix("api_error:") in MEMORY_API_ERROR_TYPES


def normalize_memory_missing_reason(reason: Any) -> str:
    """Normalize source telemetry reasons into the shared closed allowlist."""

    return reason if missing_reason_allowed(reason) else "invalid_api_payload"


def validate_memory_limit_contract(value: Any) -> dict[str, Any]:
    """Validate and return the exact best-effort memory-limit envelope.

    ``set_memory_limit`` is telemetry/configuration only.  The current H0
    contract never claims a hard limit, and an unavailable API is represented
    explicitly instead of being silently treated as applied.
    """

    if not isinstance(value, Mapping) or set(value) != MEMORY_LIMIT_KEYS:
        _fail("memory_limit contract is not closed")
    if any(type(value[key]) is not bool for key in ("attempted", "hard_limit", "applied")):
        _fail("memory_limit flags must be booleans")
    reason = value["missing_reason"]
    if reason is not None and not missing_reason_allowed(reason):
        _fail("memory_limit missing reason is not registered")
    if value["hard_limit"]:
        _fail("memory_limit must not claim a hard limit")
    if not value["attempted"]:
        if value["applied"] or reason != "api_unavailable":
            _fail("unattempted memory_limit must report api_unavailable")
    elif value["applied"]:
        if reason is not None:
            _fail("applied memory_limit cannot have a missing reason")
    elif reason == "invalid_source_value":
        pass
    elif not (
        isinstance(reason, str)
        and reason.startswith("api_error:")
        and reason.removeprefix("api_error:") in MEMORY_API_ERROR_TYPES
    ):
        _fail("failed memory_limit requires a registered api_error reason")
    return dict(value)


def validate_sign_invariant_case(case: Mapping[str, Any], fixture_seed: int, fixture_digest_value: str) -> None:
    """Validate the closed sign-invariant relation to the trusted fixture."""

    if not isinstance(case, Mapping) or set(case) != SIGN_INVARIANT_KEYS:
        _fail("sign-invariant case contract is not closed")
    if (
        case.get("name") != SIGN_INVARIANT_CASE_NAME
        or type(case.get("seed")) is not int
        or case.get("seed") != fixture_seed
        or case.get("fixture_digest") != fixture_digest_value
        or case.get("reference") != PERFORMANCE_CASE_NAME
        or case.get("relation") != "negate_left_operand"
    ):
        _fail("sign-invariant case is not bound to the trusted fixture relation")
    if case.get("passed") is not True:
        _fail("sign-invariant case failed")

# Values were generated once with the current local NumPy PCG64 generator,
# FP32 uniform draw, and C-contiguous little-endian FP16 conversion.  The
# formula is intentionally the producer's existing digest formula.
_DIGESTS = {
    "visible_64": (
        "86e143c328289c9df6c911018b9be1e44fe5d11797fdb9ec0b6d7ab5f919ef95",
        "efc455ca4c19acbdd3dc73fc0e5da605f94303656187c6c822583fb707dd8086",
        "8d259659ef2a0e72311d973e75bed56d181c09a10c48d1d6e18cbb9426a8de5b",
    ),
    "visible_rectangular": (
        "f8e3490dab1b9c4922fee2175cc1d4b6de9cb72c64936c6e9935af71bc2938d7",
        "3b749abcc88105db86402aacf32c9cbe770ce08a726a8ae24af4e3f917808d36",
        "4a0d78b31215735f129ac4e895b0afe5ac6879391c1415f554620be3f22e1039",
    ),
    "visible_zero_rhs": (
        "0f212455f84b2ec4f83f2c7ef48e4048546e45340512326cf059ef816d20c95d",
        "6d6f8d059962ba773b0a71c69cc1adb9fc74e941f495279eaef56f7f1ff77302",
        "2faca5b3b762e81b86bd7a734b0d9d9aacdc0570593d928967de7934c7bbd986",
    ),
    "visible_small_range": (
        "7fc3770269e276c1726d1fd1a96cd33aee853387ca9eef9fbfd6df8d4210a3a7",
        "dd58f37cccf931dda0bfdf1072d1d430c2eb2f21d40db5936e304f1930ff8ad8",
        "e1243e4e93ce585d298e278eac0c39993fc94abfa9e37be6432a8d8fa51d8869",
    ),
    "visible_large_range": (
        "3681a7c106b713f11c1211314ab22b4dc94cd7c852eefe39f500cc73744a230e",
        "b5515d09d48e4994762f0b52af5f523fc8726087e08e22d93c61e34cc1ce8a40",
        "b948c8b5ea084952202ae42774521b712309fdcbabeaea4f97a5bc67ad222e91",
    ),
    "holdout_uniform": (
        "4e402922a0955e09a1efc337d81540a22de90c88df1e40fd6a1a0533df59b077",
        "7b2fa3e0486c621125e619b66ac3ba1283fdbc242fc0bc59aa5ed769ab50f94d",
        "e990065388b83ed46e6f947a45a27eb164fa4cbbc0c542d91a56b2030d0f9c0e",
    ),
    "holdout_large_range": (
        "ffbd6687c8319920f1439cbc48c3367c5f022084853528f0240eccc4ab444ff2",
        "c8e02c8b1bedbaf7ce5de335c43c6eeb1345095f25284feeda1bf8b749d9dc54",
        "6dc54084a1985f34ab18a37e065519c2998e7d495162abe0d90c91623a1ede39",
    ),
}

# Trusted identities for the production performance fixture.  These are
# generated from the manifest's registered ``PCG64``/FP32 uniform stream and
# little-endian contiguous FP16 conversion.  Keeping all four values here is
# deliberate: a producer cannot co-mutate supplied hashes into acceptance.
_PERFORMANCE_IDENTITIES = {
    0xF17A2026: ("33043be0345487a8a41b522df292e5288914b9c6c6c4dc823dbec72b9146bf86", "dd40817873b24c2e6117e4e6eeebddccf89775bd4ee4453e7d5456a911670ac2", "1e26b28978e01ad0faaf296b48043e63803488cdb59e3aa84e79b9ab48a3bb20", "4776038d9500bad4374410fe2e4a167a6f834e80f0e4d19336592f4ff455dfa4"),
    0xF17A2126: ("80ec7f64599067c9b7220a830271fd198ec9fab4eb8c76d6f263dcf237eeab67", "eece8731833e155e2603f67511612e580270998aca10421169779252848b1dab", "a7c77354473a3e2d7ad6d5bf58ebd865b28972a9a817a678707c879cad011934", "c0c90c38c902b31e09439db9fd0a0b79195160b8591a53efaa1f19d17e01db41"),
    0xAA1A2026: ("8aa796e131333a767892d5277cab9e1a94cf9d04c5e986cc68791264d512a8e6", "6d8a8b8c66ded6aa93da2eb0248559e587e908a3780902f52a902349c9cec231", "cb256a1866514315c1718fd4fbb1481c497b08754258794b1d3e041480ed7c87", "6b20b4f8e1a0dfc447c19b982d372b171f16eb4180fc4998bd11956ce77ac15f"),
    0xAA1A2027: ("70977045651b39102bbeba4a49a863ad691adde403f603b6c7930cb1f9c46e63", "fefce5fbe1c5d892adb85a8aea5e3a05d2a9f8e8d84a71d48b29601162cddcf4", "14d2cbcebae0320cc9bf6e5bd0702a1ec6a96a7fd9915a08afafa2b8af995dde", "90d2660ae189902446d2ca21e32bd7ea2932c4fdfc9e6676ddb6cdd97aa9b215"),
    0xAA1A2028: ("d95c329397909ebf680f699aa79dda2388a039155c9c30a973b1f692c7c72556", "ea34194e1b7a46dab10eafb33fe3d420f092599cd148a0b6b7c86520d54b83b3", "9ae92e4e4fe84e6e73681ceb9ce5028a10551f1e3fa77f9da50747b1259dada5", "2ee80d38a92d2b40b9d77931eea4defc2c9a8572700c991bd52bc0a91c17889e"),
    0xAA1A2126: ("11cc9d4f8b6bbbd09cd4534f27898ea934ec67a47525f821e857aa3848cd1b20", "0d2b6bdec30fc9c6204ebf14501827725ee665c996bc627ec975375f14a2b5e1", "fa35a4f604e3df0bc741aec54431d4b0c30d96f61d3ec8684468ec2faab50f48", "ce05ac65dcef350cf1c5e8a07dea90f6b93f1c717d0539118ae8751e03544675"),
    0xAA1A2127: ("92ffdb823bcaef86c4ecb1db74c1545131a058f499259690af06d168b1542511", "b5fa5af61d6e172283918b08697652bc79f7ba1fa8762678ef09923cc60f6313", "8c09e8e9903d19828197b142200d23cb9870fadba151f9fb748204d4eb13315a", "89a7a1e20b858236910d9465150f21060228ae2b83018000c3e26142ca91c2a6"),
    0xAA1A2128: ("2e68438c02712827144b6203fde8199664698762009a3b7e735fe35600f7d713", "be5bbb8eb67c224507b488bdc09fd63dd9db8fb7c1f9e14ffcc7ade9b3538c13", "19bbe4de13c69a0bd1ab448bcb051b4765fa6f197dfecf2af4ab4fbbdac00a2c", "6ff844497d0a764f5a85bb5611f21260268fb54a3eab0313103cbe0d70c18104"),
}

PERFORMANCE_FIXTURE_KEYS = frozenset(
    {"a_shape", "b_shape", "dtype", "layout", "fixture_seed", "a_sha256", "b_sha256", "metadata_sha256", "fixture_sha256"}
)


def trusted_performance_fixture_identity(
    *, a_shape: Sequence[Any], b_shape: Sequence[Any], dtype: Any, layout: Any, fixture_seed: Any,
) -> dict[str, Any]:
    """Return the immutable registered identity for one production fixture."""

    if (
        not isinstance(a_shape, Sequence) or isinstance(a_shape, (str, bytes, bytearray))
        or not isinstance(b_shape, Sequence) or isinstance(b_shape, (str, bytes, bytearray))
        or list(a_shape) != [2048, 2048] or list(b_shape) != [2048, 2048]
        or dtype != "float16" or layout != "C-contiguous"
        or type(fixture_seed) is not int or fixture_seed not in _PERFORMANCE_IDENTITIES
    ):
        _fail("performance fixture identity is not registered")
    a_sha256, b_sha256, metadata_sha256, fixture_sha256 = _PERFORMANCE_IDENTITIES[fixture_seed]
    return {
        "a_shape": [2048, 2048], "b_shape": [2048, 2048], "dtype": "float16", "layout": "C-contiguous",
        "fixture_seed": fixture_seed, "a_sha256": a_sha256, "b_sha256": b_sha256,
        "metadata_sha256": metadata_sha256, "fixture_sha256": fixture_sha256,
    }


def validate_performance_fixture_identity(fixture: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any co-mutated fixture fields and return the trusted identity."""

    if not isinstance(fixture, Mapping) or set(fixture) != PERFORMANCE_FIXTURE_KEYS:
        _fail("performance fixture identity is not closed")
    expected = trusted_performance_fixture_identity(
        a_shape=fixture["a_shape"], b_shape=fixture["b_shape"], dtype=fixture["dtype"],
        layout=fixture["layout"], fixture_seed=fixture["fixture_seed"],
    )
    if dict(fixture) != expected:
        _fail("performance fixture identity is not trusted")
    return expected


def registered_digests(name: str) -> tuple[str, str, str]:
    """Return the immutable registered ``a``, ``b``, and composite digests."""

    try:
        return _DIGESTS[name]
    except KeyError as exc:
        raise CorrectnessContractError(f"unknown correctness case: {name}") from exc


def fixture_digest(a_sha256: str, b_sha256: str, seed: int) -> str:
    """Return the registered digest binding the exact fixture inputs."""

    return hashlib.sha256((a_sha256 + b_sha256 + str(seed)).encode("utf-8")).hexdigest()


def _fail(message: str) -> None:
    raise CorrectnessContractError(message)


def validate_fixed_case(case: Mapping[str, Any]) -> None:
    """Validate a generated fixed case's exact metadata and input digests."""

    if not isinstance(case, Mapping):
        _fail("correctness case is not an object")
    if set(case) != CORRECTNESS_FULL_KEYS:
        _fail("correctness case contract is not closed")
    name = case.get("name")
    if name not in _DIGESTS:
        _fail("correctness case is not registered")
    expected = next(item for item in CORRECTNESS_CASES if item[0] == name)
    expected_shape = [*expected[1], *expected[2]]
    a_sha256, b_sha256, digest = _DIGESTS[name]
    if (
        case.get("shape") != expected_shape
        or case.get("dtype") != "float16"
        or case.get("layout") != "C-contiguous"
        or case.get("seed") != expected[3]
        or case.get("zero_rhs") is not expected[6]
        or case.get("a_sha256") != a_sha256
        or case.get("b_sha256") != b_sha256
        or case.get("fixture_digest") != digest
        or case.get("hard_caps") != CORRECTNESS_HARD_CAPS
    ):
        _fail(f"correctness case {name} is not the registered fixture")
    if fixture_digest(case["a_sha256"], case["b_sha256"], case["seed"]) != case["fixture_digest"]:
        _fail(f"correctness case {name} fixture digest is not bound to its inputs")


def validate_performance_case(case: Mapping[str, Any], fixture: Mapping[str, Any]) -> None:
    """Validate the dynamic performance case against the generated fixture."""

    if not isinstance(case, Mapping) or set(case) != CORRECTNESS_FULL_KEYS or case.get("name") != PERFORMANCE_CASE_NAME:
        _fail("performance correctness case is not registered")
    trusted = validate_performance_fixture_identity(fixture)
    expected_shape = [*trusted["a_shape"], *trusted["b_shape"]]
    if (
        case.get("shape") != expected_shape
        or case.get("dtype") != trusted["dtype"]
        or case.get("layout") != trusted["layout"]
        or case.get("seed") != trusted["fixture_seed"]
        or case.get("zero_rhs") is not False
        or case.get("a_sha256") != trusted["a_sha256"]
        or case.get("b_sha256") != trusted["b_sha256"]
        or case.get("hard_caps") != CORRECTNESS_HARD_CAPS
        or fixture_digest(case["a_sha256"], case["b_sha256"], case["seed"]) != case.get("fixture_digest")
    ):
        _fail("performance correctness case is not bound to the fixture")


def validate_digest_shape(case: Mapping[str, Any]) -> None:
    """Validate digest syntax before applying a fixed/dynamic binding."""

    for field in ("a_sha256", "b_sha256", "fixture_digest"):
        if not isinstance(case.get(field), str) or _SHA256.fullmatch(case[field]) is None:
            _fail(f"correctness.{field} is not a lowercase SHA-256")
