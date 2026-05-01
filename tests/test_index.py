from pathlib import Path

from lcp_utils.parser.index import IndexEntry, dump, file_digest, load

RESOURCE = Path(__file__).parent.parent / "test_resources" / "mock_index.dat"


def test_load_index_dat() -> None:
    index = load(RESOURCE.read_bytes())

    assert index.version == 3
    assert len(index.entries) == 2

    entry = index.entries[0]
    assert entry.path == "/profiles/sample_body_sample_lens_raw.lcp"
    assert entry.identifier == 0x0102030405060708
    assert entry.exif_make == "Fixture Maker"
    assert entry.lens_pretty_name == "Sample Lens 18-55"
    assert entry.alt_lens_ids == ["SampleAltId"]
    assert entry.alt_lens_names == [
        "Sample Lens Alias A",
        "Sample Lens Alias B",
    ]


def test_dump_index_dat_round_trips_unchanged() -> None:
    source = RESOURCE.read_bytes()

    assert dump(load(source)) == source


def test_dump_modified_index() -> None:
    index = load(RESOURCE.read_bytes())

    index.entries[0].profile_name = "Modified Profile"
    index.entries.append(
        IndexEntry(
            path="/tmp/example.lcp",
            identifier=1,
            file_name="example.lcp",
            profile_name="Example",
        )
    )

    reloaded = load(dump(index))

    assert reloaded.entries[0].profile_name == "Modified Profile"
    assert reloaded.entries[-1] == index.entries[-1]


def test_file_digest_normalizes_line_endings() -> None:
    text = b"<profile>\r\n <name>Fixture</name>\r\n</profile>\r\n"

    assert file_digest(text) == "9B56D9AADE923BCF20FFC0BA4C4C2C53"
    assert file_digest(text) == file_digest(text.replace(b"\r\n", b"\n"))
