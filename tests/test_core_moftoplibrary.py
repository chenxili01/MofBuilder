import json
from pathlib import Path

from mofbuilder.core.moftoplibrary import MofTopLibrary


def test_read_mof_top_dict_populates_dictionary():
    lib = MofTopLibrary()
    lib._read_mof_top_dict("tests/database")

    assert lib.mof_top_dict is not None
    assert "UiO-66" in lib.mof_top_dict
    assert "Zr" in lib.mof_top_dict["UiO-66"]["metal"]
    assert lib.mof_top_dict["UiO-66"]["role_metadata"] is None


def test_read_mof_top_dict_loads_optional_role_metadata_sidecar(tmp_path):
    db = tmp_path / "db"
    (db / "template_database").mkdir(parents=True)
    (db / "MOF_topology_dict").write_text(
        "MOF            node_connectivity    metal     linker_topic     topology \n"
        "TEST-MULTI             8             Zn           4              csq\n",
        encoding="utf-8",
    )
    (db / "MOF_topology_role_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "families": {
                    "TEST-MULTI": {
                        "node_roles": [
                            {
                                "role_id": "node:cluster",
                                "expected_connectivity": 8,
                                "topology_labels": ["V_A", "V_A"],
                            },
                            {
                                "role_id": "node:porphyrin",
                                "expected_connectivity": 4,
                                "topology_labels": ["V_B"],
                            },
                        ],
                        "edge_roles": [
                            {
                                "role_id": "edge:ditopic",
                                "linker_connectivity": 2,
                                "topology_labels": ["EC_A"],
                            },
                            {
                                "role_id": "edge:tetratopic",
                                "linker_connectivity": 4,
                                "topology_labels": ["EC_B"],
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    lib = MofTopLibrary()
    lib._read_mof_top_dict(str(db))

    assert lib.mof_top_dict["TEST-MULTI"]["role_metadata"] == {
        "schema": "mof_topology_role_metadata/v1",
        "node_roles": [
            {
                "role_id": "node:cluster",
                "expected_connectivity": 8,
                "topology_labels": ["V_A"],
            },
            {
                "role_id": "node:porphyrin",
                "expected_connectivity": 4,
                "topology_labels": ["V_B"],
            },
        ],
        "edge_roles": [
            {
                "role_id": "edge:ditopic",
                "linker_connectivity": 2,
                "topology_labels": ["EC_A"],
            },
            {
                "role_id": "edge:tetratopic",
                "linker_connectivity": 4,
                "topology_labels": ["EC_B"],
            },
        ],
    }
    assert lib.get_role_metadata("TEST-MULTI") == lib.mof_top_dict["TEST-MULTI"][
        "role_metadata"
    ]


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


def test_fetch_preserves_legacy_scalars_and_selected_role_metadata(tmp_path):
    db = tmp_path / "db"
    (db / "template_database").mkdir(parents=True)
    (db / "template_database" / "csq.cif").write_text("data_test\n", encoding="utf-8")
    (db / "MOF_topology_dict").write_text(
        "MOF            node_connectivity    metal     linker_topic     topology \n"
        "TEST-MULTI             8             Zn           4              csq\n",
        encoding="utf-8",
    )
    (db / "MOF_topology_role_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "families": {
                    "TEST-MULTI": {
                        "node_roles": [
                            {
                                "role_id": "node:cluster",
                                "expected_connectivity": 8,
                                "topology_labels": ["V_A"],
                            }
                        ],
                        "edge_roles": [
                            {
                                "role_id": "edge:tetratopic",
                                "linker_connectivity": 4,
                                "topology_labels": ["EC_B"],
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    lib = MofTopLibrary()
    lib.data_path = str(db)

    out = lib.fetch("TEST-MULTI")

    assert out == str(db / "template_database" / "csq.cif")
    assert lib.node_connectivity == 8
    assert lib.linker_connectivity == 4
    assert lib.role_metadata == {
        "schema": "mof_topology_role_metadata/v1",
        "node_roles": [
            {
                "role_id": "node:cluster",
                "expected_connectivity": 8,
                "topology_labels": ["V_A"],
            }
        ],
        "edge_roles": [
            {
                "role_id": "edge:tetratopic",
                "linker_connectivity": 4,
                "topology_labels": ["EC_B"],
            }
        ],
    }
