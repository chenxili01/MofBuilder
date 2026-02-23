from pathlib import Path

from mofbuilder.md.gmxfilemerge import GromacsForcefieldMerger


def test_backup_and_rename_nonempty_directory(tmp_path):
    merger = GromacsForcefieldMerger()

    target = tmp_path / "MD_run" / "itps"
    target.mkdir(parents=True)
    (target / "a.itp").write_text("dummy\n", encoding="utf-8")

    merger._backup_and_rename(str(target))

    backups = [p for p in target.parent.iterdir() if p.name.startswith("#")]
    assert target.exists()
    assert target.is_dir()
    assert len(backups) == 1


def test_get_unique_atomtypes_keeps_first_occurrence():
    merger = GromacsForcefieldMerger()

    all_secs = [
        "CA   6  12.0\n",
        "CB   6  13.0\n",
        "CA   6  99.0\n",
    ]

    unique = merger._get_unique_atomtypes(all_secs)

    assert unique == ["CA   6  12.0\n", "CB   6  13.0\n"]


def test_parsetop_splits_sections(tmp_path):
    merger = GromacsForcefieldMerger()

    top_file = tmp_path / "template.top"
    top_file.write_text(
        "[ defaults ]\n"
        "1 2 yes\n\n"
        "[ atomtypes ]\n"
        "CA  12.0\n\n"
        "[ system ]\n"
        "MOF\n\n"
        "[ molecules ]\n"
        "RES 1\n",
        encoding="utf-8",
    )

    middle, names = merger._parsetop(str(top_file))

    assert len(middle) == 4
    assert "[ defaults ]" in names[0]
    assert "[ molecules ]" in names[-1]


def test_generate_top_file_writes_model_top(tmp_path):
    merger = GromacsForcefieldMerger()

    database_dir = Path("tests/database")
    itp_dir = tmp_path / "itps"
    itp_dir.mkdir(parents=True)
    (itp_dir / "test.itp").write_text(
        "[ atomtypes ]\n"
        "; a\n"
        "AA    1   12.0\n"
        "[ moleculetype ]\n"
        "; name nrexcl\n"
        "TST 3\n",
        encoding="utf-8",
    )

    merger.database_dir = str(database_dir)
    merger.target_dir = str(tmp_path)

    top_path = merger._generate_top_file(
        itp_path=str(itp_dir),
        data_path=str(database_dir),
        res_info={"TST": 2},
        model_name="unit_mof",
    )

    text = top_path.read_text(encoding="utf-8")
    assert top_path.name == "unit_mof.top"
    assert '#include "itps/test.itp"' in text
    assert "TST" in text


def test_copy_mdps_copies_parameter_files(tmp_path):
    merger = GromacsForcefieldMerger()

    merger.target_dir = str(tmp_path)
    copied_path = merger._copy_mdps(data_path="tests/database")

    copied = sorted(p.name for p in copied_path.glob("*.mdp"))
    assert "em.mdp" in copied
    assert "npt.mdp" in copied
