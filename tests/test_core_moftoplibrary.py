from pathlib import Path

from mofbuilder.core.moftoplibrary import MofTopLibrary


def test_read_mof_top_dict_populates_dictionary():
    lib = MofTopLibrary()
    lib._read_mof_top_dict("tests/database")

    assert lib.mof_top_dict is not None
    assert "UiO-66" in lib.mof_top_dict
    assert "Zr" in lib.mof_top_dict["UiO-66"]["metal"]


def test_submit_template_updates_dictionary_and_file(tmp_path):
    db = tmp_path / "db"
    (db / "template_database").mkdir(parents=True)
    (db / "MOF_topology_dict").write_text(
        "MOF            node_connectivity    metal     linker_topic     topology \n",
        encoding="utf-8",
    )
    tpl = tmp_path / "newtop.cif"
    tpl.write_text("data_test\n", encoding="utf-8")

    lib = MofTopLibrary()
    lib.data_path = str(db)
    lib.mof_top_dict = {}

    out = lib.submit_template(
        template_cif=str(tpl),
        mof_family="TEST-MOF",
        template_mof_node_connectivity=6,
        template_node_metal="Zn",
        template_linker_topic=2,
        overwrite=True,
    )

    assert out is not None
    assert "TEST-MOF" in lib.mof_top_dict
    assert "TEST-MOF" in (db / "MOF_topology_dict").read_text(encoding="utf-8")


def test_fetch_without_family_returns_none():
    lib = MofTopLibrary()
    lib.data_path = "tests/database"

    out = lib.fetch()

    assert out is None
