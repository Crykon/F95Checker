import tempfile
import types
import unittest
import struct
from unittest import mock
from pathlib import Path

from modules.game_version import (
    GameEngine,
    detect_engine,
    detect_installed_version,
    local_version_is_outdated,
    local_version_mismatches,
    version_sort_key,
)


class GameVersionTests(unittest.TestCase):
    @staticmethod
    def make_pck(project=b'[application]\nconfig/version="0.6.2"\n', flags=2, path=b"project.godot"):
        header_size = 4 + 4 * 5 + 8 + 8 + 64
        file_base = header_size
        directory_offset = file_base + len(project)
        header = b"GDPC" + struct.pack(
            "<IIIIIQQ", 3, 4, 2, 0, flags, file_base, directory_offset
        ) + b"\0" * 64
        entry = struct.pack("<I", 1)
        entry += struct.pack("<I", len(path)) + path + b"\0" * ((-len(path)) % 4)
        entry += struct.pack("<QQ", 0, len(project)) + b"\0" * 16 + struct.pack("<I", 0)
        return header + project + entry

    def test_engine_stage_identifies_godot_from_valid_pck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            (root / "Game.pck").write_bytes(self.make_pck())
            self.assertEqual(detect_engine(executable), GameEngine.Godot)
            result = detect_installed_version(executable)
        self.assertEqual(result.engine, GameEngine.Godot)
        self.assertEqual(result.version, "0.6.2")
        self.assertEqual(result.source, "PCK/project.godot application/config/version")

    def test_binary_project_settings_are_read_from_pck(self):
        key = b"application/config/version"
        value = b"\x04\0\0\0" + struct.pack("<I", 6) + b"0.11.0" + b"\0\0"
        binary = b"ECFG" + struct.pack("<I", 1) + struct.pack("<I", len(key)) + key
        binary += struct.pack("<I", len(value)) + value
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            (root / "Game.pck").write_bytes(self.make_pck(binary, path=b"project.binary"))
            result = detect_installed_version(executable)
        self.assertEqual(result.engine, GameEngine.Godot)
        self.assertEqual(result.version, "0.11.0")
        self.assertEqual(result.source, "PCK/project.binary application/config/version")

    def test_loose_godot_project_is_read_without_pck(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            (root / "project.godot").write_text(
                '[application]\nconfig/name="Game"\nconfig/version="Release Candidate 2"\n',
                encoding="utf-8",
            )
            result = detect_installed_version(executable)
        self.assertEqual(result.engine, GameEngine.Godot)
        self.assertEqual(result.version, "Release Candidate 2")

    def test_random_pck_is_not_godot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            (root / "unrelated.pck").write_bytes(b"not a Godot pack")
            self.assertEqual(detect_engine(executable), GameEngine.Unknown)

    def test_godot_without_version_remains_godot_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            (root / "Game.pck").write_bytes(self.make_pck(b"[application]\n"))
            result = detect_installed_version(executable)
        self.assertEqual(result.engine, GameEngine.Godot)
        self.assertIsNone(result.version)

    def test_embedded_godot_pck_is_read(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "Game.exe"
            executable.write_bytes(b"MZ" + b"\0" * 4095 + self.make_pck())
            result = detect_installed_version(executable)
        self.assertEqual(result.engine, GameEngine.Godot)
        self.assertEqual(result.version, "0.6.2")

    def test_encrypted_pck_does_not_fake_a_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            (root / "Game.pck").write_bytes(self.make_pck(flags=3))
            result = detect_installed_version(executable)
        self.assertEqual(result.engine, GameEngine.Godot)
        self.assertIsNone(result.version)
        self.assertIn("Encrypted", result.error)

    def test_numeric_versions_are_comparable_independent_of_link_order(self):
        self.assertGreater(version_sort_key("1.7.1"), version_sort_key("1.4.0"))
        self.assertGreater(version_sort_key("17.1"), version_sort_key("1.4.0"))
        self.assertIsNone(version_sort_key("Chapter 3"))
        self.assertTrue(local_version_is_outdated("1.7.1", "1.4.0"))
        self.assertFalse(local_version_is_outdated("1.4.0", "1.7.1"))
        self.assertTrue(local_version_is_outdated("Chapter 4", "Chapter 3"))
        self.assertTrue(local_version_mismatches("0.9", "1.0"))
        self.assertFalse(local_version_mismatches("v1.0", "1.0"))

    def test_reads_only_player_settings_bundle_version(self):
        class Object:
            def __init__(self, name, bundle_version=None):
                self.type = types.SimpleNamespace(name=name)
                self.bundle_version = bundle_version

            def parse_as_object(self):
                return types.SimpleNamespace(bundleVersion=self.bundle_version)

            def parse_as_dict(self):
                return {"bundleVersion": self.bundle_version}

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[
                Object("GameManager", "2021.3.0f1"),
                Object("PlayerSettings", " 0.8.7 "),
            ])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "0.8.7")
        self.assertEqual(result.source, "PlayerSettings.bundleVersion")
        self.assertEqual(result.engine, "Unity")

    def test_uses_player_settings_typetree_when_object_parser_fails(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                raise ValueError("generated class unavailable")

            def parse_as_dict(self):
                return {"bundleVersion": "0.7.1"}

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "0.7.1")

    def test_uses_conservative_raw_player_settings_fallback(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                raise ValueError("unsupported Unity layout")

            def parse_as_dict(self):
                raise ValueError("unsupported Unity layout")

            def get_raw_data(self):
                return b"".join(struct.pack("<I", len(value)) + value.encode() + b"\0\0\0\0"
                               for value in ("Company", "Game", "0.90.10"))

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "0.90.10")
        self.assertEqual(result.source, "PlayerSettings.bundleVersion (raw fallback)")
        self.assertEqual(result.confidence, 0.4)

    def test_raw_player_settings_supports_alpha_versions(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                raise ValueError("unsupported Unity layout")

            def parse_as_dict(self):
                raise ValueError("unsupported Unity layout")

            def get_raw_data(self):
                value = "1.0.6a"
                return struct.pack("<I", len(value)) + value.encode()

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "1.0.6a")
        self.assertEqual(result.source, "PlayerSettings.bundleVersion (raw fallback)")

    def test_raw_player_settings_supports_prefixed_versions(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                raise ValueError("unsupported Unity layout")

            def parse_as_dict(self):
                raise ValueError("unsupported Unity layout")

            def get_raw_data(self):
                value = "beta 0.88.2"
                return struct.pack("<I", len(value)) + value.encode()

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "beta 0.88.2")

    def test_raw_player_settings_supports_marked_versions(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                raise ValueError("unsupported Unity layout")

            def parse_as_dict(self):
                raise ValueError("unsupported Unity layout")

            def get_raw_data(self):
                value = "SEASON 5 VERSION 1.07.3C"
                return struct.pack("<I", len(value)) + value.encode()

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "1.07.3C")

    def test_uses_monobehaviour_version_when_player_settings_is_default(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                return types.SimpleNamespace(bundleVersion="1.0")

            def get_raw_data(self):
                return b""

        class MonoBehaviourObject:
            type = types.SimpleNamespace(name="MonoBehaviour")

            def get_raw_data(self):
                value = "v0.1.58"
                return b"".join((b"\0" * 20, struct.pack("<I", len(value)), value.encode()))

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject(), MonoBehaviourObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(executable)
        self.assertEqual(result.version, "v0.1.58")
        self.assertEqual(result.source, "MonoBehaviour raw version field")
        self.assertEqual(result.confidence, 0.25)

    def test_non_unity_path_is_unknown(self):
        with tempfile.TemporaryDirectory() as temporary:
            executable = Path(temporary) / "Game.exe"
            executable.touch()
            result = detect_installed_version(executable)
            self.assertIsNone(result.version)
            self.assertIsNone(result.engine)

    def test_installation_root_directory_is_accepted(self):
        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").touch()
            result = detect_installed_version(root)
        self.assertEqual(result.engine, "Unity")
        self.assertIsNone(result.version)

    def test_data_unity3d_is_an_alternate_unity_source(self):
        class PlayerSettingsObject:
            type = types.SimpleNamespace(name="PlayerSettings")

            def parse_as_object(self):
                return types.SimpleNamespace(bundleVersion="0.5.7")

        fake_unitypy = types.SimpleNamespace(
            load=lambda _: types.SimpleNamespace(objects=[PlayerSettingsObject()])
        )
        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            "sys.modules", {"UnityPy": fake_unitypy}
        ):
            root = Path(temporary)
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "data.unity3d").touch()
            result = detect_installed_version(root)
        self.assertEqual(result.version, "0.5.7")

    def test_unity_signature_is_detected_without_binary_fixture(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            executable = root / "Game.exe"
            executable.touch()
            data_dir = root / "Game_Data"
            data_dir.mkdir()
            (data_dir / "globalgamemanagers").write_bytes(b"not a serialized Unity file")
            result = detect_installed_version(executable)
            self.assertIsNone(result.version)
            self.assertEqual(result.engine, "Unity")
            self.assertTrue(
                "parse Unity data" in result.error or
                "PlayerSettings.bundleVersion not found" in result.error
            )


if __name__ == "__main__":
    unittest.main()
