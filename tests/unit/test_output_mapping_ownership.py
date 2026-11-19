# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Output ownership includes the model instance and selected components."""

from types import SimpleNamespace

import numpy as np
import pytest

from ovfmi import MissingInputPolicy
from ovfmi._parser import FmuDirection, FmuParserConnection, FmuParserMapping
from ovfmi.routing import InputRouter, OutputRouter


def connection(selector=(0, 1), name="force_x", target="/State", attribute="output",
               enabled=True, direction=FmuDirection.OUTPUT):
    return FmuParserConnection(enabled, [target], [
        FmuParserMapping(name, attribute, direction, selector),
    ])


def instance(path, *connections):
    parsed = SimpleNamespace(path=path, connections=list(connections))
    return SimpleNamespace(get_parser_instance=lambda: parsed)


@pytest.fixture
def router():
    inputs = InputRouter({}, MissingInputPolicy.USE_FMU_START_VALUE)
    return OutputRouter({}, inputs, strict=True)


@pytest.mark.parametrize("second_name", ["force_x", "force_y"])
@pytest.mark.parametrize("selectors", [
    ((0, 1), (0, 1)),
    ((2, 3), (4, 2)),
    ((4, 2), (2, 3)),
    ((1, 5), (2, 1)),
    ((2, 1), (1, 5)),
    ((0, 0), (0, 0)),
    ((0, 0), (5, 1)),
    ((5, 1), (0, 0)),
])
def test_distinct_instances_cannot_overlap(router, selectors, second_name):
    first, second = selectors
    router.cache_connections(instance("/Model0", connection(first)))
    with pytest.raises(ValueError, match="overlapping output mapping for /State.output"):
        router.cache_connections(instance("/Model1", connection(second, second_name)))


@pytest.mark.parametrize("selectors", [((0, 1), (1, 1)), ((1, 2), (3, 2)), ((5, 1), (1, 2))])
def test_disjoint_components_are_accepted(router, selectors):
    for index, selector in enumerate(selectors):
        model = instance(f"/Model{index}", connection(selector))
        router.cache_connections(model)
        assert router.get_outputs_for(model) == ["force_x"]


def test_disjoint_registration_comparisons_grow_at_most_quadratically(router):
    class Component(int):
        comparisons = 0

        def __lt__(self, other):
            Component.comparisons += 1
            return super().__lt__(other)

    count = 100
    for index in range(count):
        router.cache_connections(instance(f"/Model{index}", connection((Component(index), 1))))
    assert 0 < Component.comparisons < count * count


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
def test_disjoint_producers_keep_both_values(router, order):
    models = [instance(f"/Model{i}", connection((i, 1))) for i in range(2)]
    for model in models:
        router.cache_connections(model)
    for index in order:
        result = np.array([(-6.25 if index == 0 else -25.0,)], dtype=[("force_x", np.float64)])[0]
        router.write_outputs(models[index], [], result)
    assert router.snapshot()[0].tensors[0].tolist() == [[-6.25, -25.0]]


@pytest.mark.parametrize("other", [connection(target="/Other"), connection(attribute="other")])
def test_distinct_targets_are_accepted(router, other):
    router.cache_connections(instance("/Model0", connection()))
    router.cache_connections(instance("/Model1", other))


@pytest.mark.parametrize("selectors", [((0, 1), (0, 1)), ((1, 3), (2, 2)), ((0, 0), (5, 1))])
def test_same_instance_and_variable_may_repeat(router, selectors):
    model = instance("/Model", *(connection(selector) for selector in selectors))
    router.cache_connections(model)
    router.cache_connections(model)


@pytest.mark.parametrize("selectors", [((0, 1), (0, 1)), ((0, 0), (5, 1)), ((5, 1), (0, 0))])
def test_different_variables_in_one_instance_cannot_overlap(router, selectors):
    with pytest.raises(ValueError, match="overlapping output mapping"):
        router.cache_connections(instance("/Model", connection(selectors[0]),
                                          connection(selectors[1], "force_y")))


@pytest.mark.parametrize("ignored", [connection(enabled=False), connection(direction=FmuDirection.INPUT)])
def test_inactive_outputs_do_not_claim_components(router, ignored):
    router.cache_connections(instance("/Model0", ignored))
    router.cache_connections(instance("/Model1", connection(name="force_y")))


def test_removing_or_replacing_mappings_releases_ownership(router):
    first = instance("/Model0", connection())
    router.cache_connections(first)
    router.empty_cache_for(first)
    router.cache_connections(instance("/Model1", connection(name="force_y")))
    router.cache_connections(instance("/Model1", connection((1, 1), "force_y")))
    router.cache_connections(first)


def test_failed_registration_does_not_claim_components(router):
    router.cache_connections(instance("/Model0", connection()))
    with pytest.raises(ValueError, match="overlapping output mapping"):
        router.cache_connections(instance("/Model1", connection((1, 1)),
                                          connection(name="force_y")))
    router.cache_connections(instance("/Model2", connection((1, 1), "force_y")))


def test_failed_replacement_preserves_previous_mappings(router):
    first = instance("/Model0", connection())
    second = instance("/Model1", connection((1, 1), "force_y"))
    router.cache_connections(first)
    router.cache_connections(second)
    with pytest.raises(ValueError, match="overlapping output mapping"):
        router.cache_connections(instance("/Model1", connection()))
    assert router.get_outputs_for(second) == ["force_y"]
    with pytest.raises(ValueError, match="overlapping output mapping"):
        router.cache_connections(instance("/Model2", connection((1, 1))))


def test_reset_preserves_ownership(router):
    router.cache_connections(instance("/Model0", connection()))
    router.reset()
    with pytest.raises(ValueError, match="overlapping output mapping"):
        router.cache_connections(instance("/Model1", connection()))


@pytest.mark.parametrize("selectors", [((0, 1), (0, 1)), ((0, 0), (5, 1)), ((5, 1), (0, 0))])
def test_nonstrict_overlap_is_accepted(router, selectors):
    router._strict = False
    for index, selector in enumerate(selectors):
        router.cache_connections(instance(f"/Model{index}", connection(selector)))
