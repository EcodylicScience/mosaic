

def test_two_recipes_for_one_entry_are_reported_rather_than_refused(
    scenario_dataset: Dataset,
) -> None:
    """Found on four real datasets carrying both a conversion and a tracker run.

    ``select_variant_rows`` raises when one entry has two genuine tracks
    recipes, which is right for *executing*: there is no defensible default and
    a guess reads the wrong table. An inventory is not executing. Refusing turns
    a describable dataset into an exception and takes the report of every other
    artifact down with it, which is the opposite of what a reader asking "what
    is in here" needs.
    """
    from mosaic.core.pipeline.inventory.scan import reportable_universe

    from tests.conftest import add_tracks_variant

    add_tracks_variant(scenario_dataset, "convert-x.0.1-aaaaaaaaaa", "seq_a")
    add_tracks_variant(scenario_dataset, "trex.0.1-bbbbbbbbbb", "seq_a")

    with pytest.raises(ValueError, match="variants"):
        _ = entry_universe(scenario_dataset)

    universe = reportable_universe(scenario_dataset)

    assert ("", "seq_a") in universe
    found = inventory(scenario_dataset, kinds=["feature", "tracks-variant"])
    assert found.records
