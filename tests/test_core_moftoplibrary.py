import json

import pytest

from mofbuilder.core.moftoplibrary import MofTopLibrary


def _canonical_family_role_metadata():
    return {
        "schema_name": "mof_reticular_role_metadata",
        "schema_version": 1,
        "family_name": "TEST-MULTI",
        "roles": {
            "VA": {"role_class": "V", "canonical_role_id": "node:VA"},
            "CA": {"role_class": "C", "canonical_role_id": "node:CA"},
            "EA": {"role_class": "E", "canonical_role_id": "edge:EA"},
            "EB": {"role_class": "E", "canonical_role_id": "edge:EB"},
        },
        "connectivity_rules": {
            "VA": {"incident_edge_aliases": ["EA", "EA", "EB", "EB"]},
            "CA": {"incident_edge_aliases": ["EA", "EA"]},
        },
        "path_rules": [
            {"edge_alias": "EA", "endpoint_pattern": ["VA", "EA", "CA"]},
            {"edge_alias": "EB", "endpoint_pattern": ["VA", "EB", "VA"]},
        ],
        "bundle_rules": {
            "CA": {
                "bundle_owner": "linker",
                "attachment_edge_aliases": ["EA", "EA"],
            }
        },
        "slot_rules": {
            "VA": [
                {"attachment_index": 0, "slot_type": "XA"},
                {"attachment_index": 1, "slot_type": "XA"},
                {"attachment_index": 2, "slot_type": "XB"},
                {"attachment_index": 3, "slot_type": "XB"},
            ],
            "CA": [
                {"attachment_index": 0, "slot_type": "XA"},
                {"attachment_index": 1, "slot_type": "XA"},
            ],
            "EA": [
                {"attachment_index": 0, "slot_type": "XA", "endpoint_side": "V"},
                {"attachment_index": 1, "slot_type": "XA", "endpoint_side": "C"},
            ],
            "EB": [
                {"attachment_index": 0, "slot_type": "XB", "endpoint_side": "V"},
                {"attachment_index": 1, "slot_type": "XB", "endpoint_side": "V"},
            ],
        },
        "cyclic_order_rules": {
            "CA": {
                "ordered_attachment_indices": [0, 1],
                "order_kind": "clockwise_local_topology",
            }
        },
        "edge_kind_rules": {
            "EA": {"edge_kind": "real"},
            "EB": {
                "edge_kind": "null",
                "null_payload_model": "duplicated_zero_length_anchors",
            },
        },
        "resolve_rules": {
            "EA": {"resolve_mode": "ownership_transfer"},
            "EB": {"resolve_mode": "alignment_only"},
        },
        "unresolved_edge_policy": {
            "default_action": "error",
            "allowed_null_fallback_edge_aliases": ["EB"],
        },
        "fragment_lookup_hints": {
            "VA": {"library": "nodes_database", "keywords": ["2c", "rod", "Al"]},
            "CA": {"library": "linker_input", "fragment_kind": "center"},
            "EA": {"library": "linker_input", "fragment_kind": "connector"},
            "EB": {"library": "family_metadata", "fragment_kind": "null_edge"},
        },
    }


def _compat_family_role_metadata():
    return {
        "schema": "mof_topology_role_metadata/v1",
        "node_roles": [
            {
                "role_id": "node:VA",
                "expected_connectivity": 4,
                "topology_labels": ["VA"],
            },
            {
                "role_id": "node:CA",
                "expected_connectivity": 2,
                "topology_labels": ["CA"],
            },
        ],
        "edge_roles": [
            {
                "role_id": "edge:EA",
                "linker_connectivity": 2,
                "topology_labels": ["EA"],
            },
            {
                "role_id": "edge:EB",
                "linker_connectivity": 2,
                "topology_labels": ["EB"],
            },
        ],
        "canonical_role_metadata": _canonical_family_role_metadata(),
    }


def _write_test_database(db_path, *, metadata=None):
    (db_path / "template_database").mkdir(parents=True)
    (db_path / "MOF_topology_dict").write_text(
        "MOF            node_connectivity    metal     linker_topic     topology \n"
        "TEST-MULTI             8             Zn           4              csq\n",
        encoding="utf-8",
    )
    (db_path / "template_database" / "csq.cif").write_text(
        "data_test\n",
        encoding="utf-8",
    )
    if metadata is not None:
        (db_path / "MOF_topology_role_metadata.json").write_text(
            json.dumps(
                {
                    "schema_name": "mof_reticular_role_metadata",
                    "schema_version": 1,
                    "families": {"TEST-MULTI": metadata},
                }
            ),
            encoding="utf-8",
        )


def test_read_mof_top_dict_keeps_legacy_family_scalars_without_role_metadata():
    lib = MofTopLibrary()
    lib._read_mof_top_dict("tests/database")

    assert lib.mof_top_dict is not None
    assert "UiO-66" in lib.mof_top_dict
    assert lib.mof_top_dict["UiO-66"]["node_connectivity"] == 12
    assert lib.mof_top_dict["UiO-66"]["linker_topic"] == 2
    assert "Zr" in lib.mof_top_dict["UiO-66"]["metal"]
    assert lib.mof_top_dict["UiO-66"]["role_metadata"] is None


def test_read_mof_top_dict_without_sidecar_keeps_role_metadata_none(tmp_path):
    db = tmp_path / "db"
    _write_test_database(db)

    lib = MofTopLibrary()
    lib._read_mof_top_dict(str(db))

    assert lib.mof_top_dict["TEST-MULTI"]["node_connectivity"] == 8
    assert lib.mof_top_dict["TEST-MULTI"]["linker_topic"] == 4
    assert lib.mof_top_dict["TEST-MULTI"]["role_metadata"] is None


def test_read_mof_top_dict_loads_builder_compatible_role_metadata_sidecar(tmp_path):
    db = tmp_path / "db"
    expected_role_metadata = _compat_family_role_metadata()
    _write_test_database(db, metadata=expected_role_metadata["canonical_role_metadata"])

    lib = MofTopLibrary()
    lib._read_mof_top_dict(str(db))

    assert lib.mof_top_dict["TEST-MULTI"]["role_metadata"] == expected_role_metadata
    assert lib.get_role_metadata("TEST-MULTI") == expected_role_metadata
    assert (
        lib.get_canonical_role_metadata("TEST-MULTI")
        == expected_role_metadata["canonical_role_metadata"]
    )


def test_read_mof_top_dict_rejects_invalid_schema_at_library_boundary(tmp_path):
    db = tmp_path / "db"
    invalid_metadata = _canonical_family_role_metadata()
    invalid_metadata["path_rules"] = [
        {"edge_alias": "EA", "endpoint_pattern": ["CA", "EA", "VA"]}
    ]
    _write_test_database(db, metadata=invalid_metadata)

    lib = MofTopLibrary()

    with pytest.raises(ValueError, match="V-E-V or V-E-C"):
        lib._read_mof_top_dict(str(db))


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
    expected_role_metadata = _compat_family_role_metadata()
    _write_test_database(db, metadata=expected_role_metadata["canonical_role_metadata"])

    lib = MofTopLibrary()
    lib.data_path = str(db)

    out = lib.fetch("TEST-MULTI")

    assert out == str(db / "template_database" / "csq.cif")
    assert lib.node_connectivity == 8
    assert lib.linker_connectivity == 4
    assert lib.role_metadata == expected_role_metadata
    assert lib.canonical_role_metadata == expected_role_metadata["canonical_role_metadata"]
