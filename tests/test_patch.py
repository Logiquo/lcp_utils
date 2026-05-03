import json
import shutil
from pathlib import Path

from lcp_utils.parser import index
from lcp_utils.patch import INDEX_LOGICAL_ROOT, patch_profiles


def test_patch_profiles_backs_up_copies_and_updates_index(tmp_path: Path) -> None:
    source_dir = tmp_path / "profiles"
    target_dir = tmp_path / "LensProfiles"
    source_dir.mkdir()
    target_dir.mkdir()

    profile_name = "SONY (Viltrox AF 35mm F1.8 EVO) - RAW.lcp"
    install_path = "/Viltrox/Sony/SONY (Viltrox AF 35mm F1.8 EVO) - RAW.lcp"
    shutil.copy2(Path("profiles") / profile_name, source_dir / profile_name)
    (source_dir / "patch.json").write_text(
        json.dumps({profile_name: install_path}),
        encoding="utf-8",
    )
    shutil.copy2(
        Path("test_resources") / "mock_index.dat",
        target_dir / "Index.dat",
    )

    entries = patch_profiles(source_dir, target_dir)

    backups = list(target_dir.glob("Index.dat.bak.*"))
    assert len(backups) == 1
    expected_backup = (Path("test_resources") / "mock_index.dat").read_bytes()
    assert backups[0].read_bytes() == expected_backup

    copied_profile = target_dir / "1.0" / "Viltrox" / "Sony" / profile_name
    assert copied_profile.read_bytes() == (source_dir / profile_name).read_bytes()

    patched = index.load((target_dir / "Index.dat").read_bytes())
    expected_path = f"{INDEX_LOGICAL_ROOT}{install_path}"
    matching = [entry for entry in patched.entries if entry.path == expected_path]
    assert len(matching) == 1
    assert entries == matching
    source_bytes = (source_dir / profile_name).read_bytes()
    assert matching[0].file_digest == index.file_digest(source_bytes)
