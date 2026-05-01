from pathlib import Path

from lcp_utils import Profile, dump, load

RESOURCE_DIR = Path(__file__).parent.parent / "test_resources"


def read_resource(name: str) -> str:
    return (RESOURCE_DIR / name).read_text(encoding="utf-8")


def test_load_direct_perspective_model_attributes() -> None:
    profiles = load(read_resource("direct_perspective.lcp"))

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.profile_name == "Direct Perspective Profile"
    assert profile.make == "SONY"
    assert profile.camera_raw_profile is True
    assert profile.focal_length == 24
    assert profile.focus_distance == 10000
    assert profile.perspective is not None
    assert profile.perspective.version == 2
    assert profile.perspective.scale_factor == 1.020306
    assert profile.perspective.radial_distort_param1 == -0.097634
    assert profile.perspective.radial_distort_param2 == 0.067408
    assert profile.perspective.radial_distort_param3 == 0.016026
    assert profile.perspective.vignette is None


def test_load_nested_perspective_model_with_vignette() -> None:
    profiles = load(read_resource("nested_perspective_vignette.lcp"))

    assert len(profiles) == 1
    profile = profiles[0]
    assert profile.profile_name == "Nested Perspective Profile"
    assert profile.perspective is not None
    assert profile.perspective.version == 2
    assert profile.perspective.radial_distort_param1 == -0.037552
    assert profile.perspective.vignette is not None
    assert profile.perspective.vignette.param1 == -0.748281
    assert profile.perspective.vignette.param2 == 0.506711
    assert profile.perspective.vignette.param3 == -0.349576


def test_load_mixed_profile_file() -> None:
    profiles = load(read_resource("mixed_profiles.lcp"))

    assert [profile.profile_name for profile in profiles] == [
        "Mixed Direct Profile",
        "Mixed Nested Profile",
    ]
    assert profiles[0].perspective is not None
    assert profiles[0].perspective.vignette is None
    assert profiles[1].perspective is not None
    assert profiles[1].perspective.vignette is not None


def test_dump_profile_round_trips_through_load() -> None:
    profile = load(read_resource("nested_perspective_vignette.lcp"))[0]

    reloaded = load(dump(profile))

    assert len(reloaded) == 1
    assert reloaded[0] == profile


def test_dump_profile_sequence_round_trips_through_load() -> None:
    profiles = load(read_resource("mixed_profiles.lcp"))

    reloaded = load(dump(profiles))

    assert len(reloaded) == 2
    assert all(isinstance(profile, Profile) for profile in reloaded)
    assert reloaded == profiles
