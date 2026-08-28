"""Detection of versions from locally installed game files.

This module deliberately keeps locally detected versions separate from the
F95Zone/API version stored in ``Game.version``.
"""

import dataclasses
import enum
import ast
import pathlib
import re
import struct
import typing


class GameEngine(str, enum.Enum):
    Unknown = "Unknown"
    Unity = "Unity"
    Godot = "Godot"


@dataclasses.dataclass(slots=True)
class LocalVersionResult:
    version: str | None
    engine: GameEngine | None
    source: str | None
    confidence: float = 0.0
    error: str | None = None


_VERSION_STRING = re.compile(
    r"^(?:(?:alpha|beta|rc|release)\s+)?v?\d+(?:\.\d+)+(?:[A-Za-z][A-Za-z0-9._-]*)?$",
    re.IGNORECASE,
)
_MARKED_VERSION_STRING = re.compile(
    r"\bversion\s*[:#]?\s*(?P<version>v?\d+(?:\.\d+)+(?:[A-Za-z][A-Za-z0-9._-]*)?)\b",
    re.IGNORECASE,
)
_COMPARABLE_VERSION = re.compile(
    r"^v?(?P<numbers>\d+(?:\.\d+)+)(?P<suffix>[A-Za-z][A-Za-z0-9._-]*)?$",
    re.IGNORECASE,
)


def version_sort_key(version: str) -> tuple[tuple[int, ...], str] | None:
    """Return a safe comparison key for numeric dotted versions only."""
    if not isinstance(version, str):
        return None
    match = _COMPARABLE_VERSION.fullmatch(version.strip())
    if not match:
        return None
    return tuple(int(number) for number in match.group("numbers").split(".")), (match.group("suffix") or "").lower()


def local_version_is_outdated(latest: str, installed: str | None) -> bool:
    """Return whether a detected local version needs an update."""
    if not latest or not installed:
        return False
    latest_key = version_sort_key(latest)
    installed_key = version_sort_key(installed)
    if latest_key is not None and installed_key is not None:
        return latest_key > installed_key
    normalize = lambda value: value.strip().removeprefix("v").lower()
    return normalize(latest) != normalize(installed)


def local_version_mismatches(latest: str, installed: str | None) -> bool:
    """Return whether a known local version differs from the API version."""
    if not latest or not installed:
        return False
    latest_key = version_sort_key(latest)
    installed_key = version_sort_key(installed)
    if latest_key is not None and installed_key is not None:
        return latest_key != installed_key
    normalize = lambda value: value.strip().removeprefix("v").lower()
    return normalize(latest) != normalize(installed)


def _extract_raw_bundle_version(obj: typing.Any) -> str | None:
    """Extract PlayerSettings.bundleVersion from raw serialized data.

    This handles builds such as Unity 2022 games whose PlayerSettings object
    cannot be decoded by UnityPy. PlayerSettings contains length-prefixed
    strings; restricting this to the raw bytes of the already-identified
    PlayerSettings object avoids scanning arbitrary game data.
    """
    raw = obj.get_raw_data()
    if not isinstance(raw, bytes):
        raw = bytes(raw)
    candidates = set()
    marked_candidates = set()
    for offset in range(0, len(raw) - 4):
        length = struct.unpack_from("<I", raw, offset)[0]
        if not 1 <= length <= 64 or offset + 4 + length > len(raw):
            continue
        try:
            value = raw[offset + 4:offset + 4 + length].decode("utf-8")
        except UnicodeDecodeError:
            continue
        if _VERSION_STRING.fullmatch(value):
            candidates.add(value.strip())
        if match := _MARKED_VERSION_STRING.search(value):
            marked_candidates.add(match.group("version").strip())
    if marked_candidates:
        candidates = marked_candidates
    if not candidates:
        return None
    # Prefer a more specific version, but refuse ties rather than guessing.
    ranked = sorted(candidates, key=lambda value: (value.count("."), len(value)), reverse=True)
    if len(ranked) > 1 and (ranked[0].count("."), len(ranked[0])) == (ranked[1].count("."), len(ranked[1])):
        return None
    return ranked[0]


def _extract_raw_monobehaviour_version(environment: typing.Any) -> str | None:
    """Find an unambiguous version string in custom MonoBehaviour data."""
    candidates = set()
    for obj in environment.objects:
        if getattr(obj.type, "name", None) != "MonoBehaviour":
            continue
        try:
            raw = obj.get_raw_data()
            if not isinstance(raw, bytes):
                raw = bytes(raw)
            for offset in range(0, len(raw) - 4):
                length = struct.unpack_from("<I", raw, offset)[0]
                if not 1 <= length <= 64 or offset + 4 + length > len(raw):
                    continue
                try:
                    value = raw[offset + 4:offset + 4 + length].decode("utf-8").strip()
                except UnicodeDecodeError:
                    continue
                if _VERSION_STRING.fullmatch(value) and value.lower().removeprefix("v") != "1.0":
                    candidates.add(value)
        except Exception:
            continue
    if not candidates:
        return None
    ranked = sorted(candidates, key=lambda value: (value.count("."), len(value)), reverse=True)
    if len(ranked) > 1 and (ranked[0].count("."), len(ranked[0])) == (ranked[1].count("."), len(ranked[1])):
        return None
    return ranked[0]


def _unity_data_dir(executable: pathlib.Path) -> pathlib.Path | None:
    """Return the Unity data directory for an executable or install root."""
    try:
        if executable.is_dir():
            if executable.name.lower().endswith("_data"):
                return executable
            candidates = [node for node in executable.iterdir()
                          if node.is_dir() and node.name.lower().endswith("_data")]
            return candidates[0] if len(candidates) == 1 else None
        if not executable.is_file():
            return None
        data_dir = executable.with_name(f"{executable.stem}_Data")
        if data_dir.is_dir():
            return data_dir
    except (OSError, ValueError):
        pass
    return None


def _installation_root(executable: pathlib.Path) -> pathlib.Path | None:
    """Resolve the directory in which an executable's game data is installed."""
    try:
        if executable.is_dir():
            return executable
        if executable.is_file():
            return executable.parent
    except (OSError, ValueError):
        pass
    return None


def _read_project_version(data: bytes, setting: str = "config/version") -> str | None:
    """Read an application setting from Godot's INI-like text format."""
    section = None
    for raw_line in data.decode("utf-8-sig", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith((";", "#")):
            continue
        section_match = re.fullmatch(r"\[([^]]+)\]", line)
        if section_match:
            section = section_match.group(1).strip().lower()
            continue
        if section != "application":
            continue
        match = re.match(re.escape(setting) + r"\s*=\s*(.*?)(?:\s*;.*)?$", line, re.IGNORECASE)
        if not match:
            continue
        value = match.group(1).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            try:
                parsed = ast.literal_eval(value)
                value = parsed if isinstance(parsed, str) else value[1:-1]
            except (SyntaxError, ValueError):
                value = value[1:-1]
        return value.strip() or None
    return None


def _read_project_binary_setting(data: bytes, setting: str = "application/config/version") -> str | None:
    """Read a string application setting from Godot's ECFG format."""
    if len(data) < 8 or data[:4] != b"ECFG":
        raise ValueError("invalid Godot project.binary header")
    count = struct.unpack_from("<I", data, 4)[0]
    cursor = 8
    for _ in range(count):
        if cursor + 4 > len(data):
            raise ValueError("truncated Godot project.binary")
        key_length = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        key_end = cursor + key_length
        if key_end > len(data):
            raise ValueError("truncated Godot project.binary key")
        key = data[cursor:key_end].decode("utf-8", errors="strict")
        cursor = key_end
        if cursor + 4 > len(data):
            raise ValueError("truncated Godot project.binary value")
        value_length = struct.unpack_from("<I", data, cursor)[0]
        cursor += 4
        value_end = cursor + value_length
        if value_end > len(data):
            raise ValueError("truncated Godot project.binary value")
        value = data[cursor:value_end]
        cursor = value_end
        if key != setting or len(value) < 8:
            continue
        variant_type, string_length = struct.unpack_from("<II", value, 0)
        if variant_type != 4 or string_length > len(value) - 8:
            continue
        return value[8:8 + string_length].decode("utf-8", errors="strict").strip() or None
    return None


@dataclasses.dataclass(slots=True)
class _PckEntry:
    offset: int
    size: int
    encrypted: bool


@dataclasses.dataclass(slots=True)
class _PckInfo:
    entries: dict[str, _PckEntry]
    encrypted_directory: bool = False


_GODOT_PCK_MAGIC = b"GDPC"
_GODOT_PCK_VERSIONS = {1, 2, 3, 4}
_PCK_DIR_ENCRYPTED = 1
_PCK_FILE_ENCRYPTED = 1
_MAX_PCK_FILES = 2_000_000
_MAX_PROJECT_SETTINGS = 4 * 1024 * 1024
_MAX_GODOT_TEXT_RESOURCE = 2 * 1024 * 1024
_MAX_GODOT_RESOURCE_SCAN = 64 * 1024 * 1024
_GODOT_RESOURCE_VERSION_PATTERNS = (
    (re.compile(rb"version_code\s*=\s*[\"']\s*(?:version\s+)?(?P<version>v?\d+(?:\.\d+)+(?:[A-Za-z][A-Za-z0-9._-]*)?)", re.IGNORECASE), 3),
    (re.compile(rb"(?:game|app|release)[_-]?version\s*=\s*[\"']\s*(?:version\s+)?(?P<version>v?\d+(?:\.\d+)+(?:[A-Za-z][A-Za-z0-9._-]*)?)", re.IGNORECASE), 2),
    (re.compile(rb"(?:text|label)\s*=\s*[\"']\s*version\s*[:.]?\s*(?P<version>v?\d+(?:\.\d+)+(?:[A-Za-z][A-Za-z0-9._-]*)?)", re.IGNORECASE), 1),
)


def _read_u32(file: typing.BinaryIO) -> int:
    data = file.read(4)
    if len(data) != 4:
        raise ValueError("truncated Godot PCK header")
    return struct.unpack("<I", data)[0]


def _read_u64(file: typing.BinaryIO) -> int:
    data = file.read(8)
    if len(data) != 8:
        raise ValueError("truncated Godot PCK header")
    return struct.unpack("<Q", data)[0]


def _parse_godot_pck(file: typing.BinaryIO, start: int) -> _PckInfo:
    """Parse a Godot v2-v4 PCK index without loading pack contents."""
    file.seek(start)
    if file.read(4) != _GODOT_PCK_MAGIC:
        raise ValueError("Godot PCK magic not found")
    version = _read_u32(file)
    if version not in _GODOT_PCK_VERSIONS:
        raise ValueError(f"unsupported Godot PCK version {version}")
    _read_u32(file)  # Godot major
    _read_u32(file)  # Godot minor
    _read_u32(file)  # Godot patch
    flags = _read_u32(file)
    file_base = _read_u64(file)
    if version in (3, 4):
        file_base += start
        directory_offset = _read_u64(file) + start
        file.seek(64, 1)  # reserved header space
    elif version == 2:
        if flags & 2:  # PACK_REL_FILEBASE
            file_base += start
        directory_offset = file.tell() + 64
        file.seek(64, 1)  # reserved header space
    else:
        # PCK v1 has thirteen reserved 32-bit words instead of v2's sixteen.
        directory_offset = file.tell() + 52
        file.seek(52, 1)
    if flags & _PCK_DIR_ENCRYPTED:
        return _PckInfo({}, encrypted_directory=True)

    file.seek(directory_offset)
    file_count = _read_u32(file)
    if file_count > _MAX_PCK_FILES:
        raise ValueError("invalid Godot PCK file count")
    file_size = file.seek(0, 2)
    file.seek(directory_offset + 4)
    entries = {}
    for _ in range(file_count):
        path_length = _read_u32(file)
        if path_length > 1024 * 1024:
            raise ValueError("invalid Godot PCK path length")
        path_data = file.read(path_length)
        if len(path_data) != path_length:
            raise ValueError("truncated Godot PCK directory")
        padding = (-path_length) % 4
        if padding:
            file.seek(padding, 1)
        relative_offset = _read_u64(file)
        size = _read_u64(file)
        file.seek(16, 1)  # MD5
        entry_flags = _read_u32(file) if version >= 2 else 0
        actual_offset = file_base + relative_offset
        if actual_offset > file_size or size > file_size - actual_offset:
            raise ValueError("invalid Godot PCK entry bounds")
        path = path_data.rstrip(b"\0").decode("utf-8", errors="strict").replace("\\", "/").lower()
        path = path.removeprefix("res://")
        entries[path] = _PckEntry(actual_offset, size, bool(entry_flags & _PCK_FILE_ENCRYPTED))
    return _PckInfo(entries)


def _find_embedded_pck(executable: pathlib.Path) -> tuple[int, _PckInfo] | None:
    """Find and validate an embedded PCK while reading the executable in chunks."""
    chunk_size = 1024 * 1024
    try:
        with executable.open("rb") as file:
            position = 0
            carry = b""
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    return None
                chunk_end = position + len(chunk)
                haystack = carry + chunk
                base = position - len(carry)
                search = 0
                while (index := haystack.find(_GODOT_PCK_MAGIC, search)) >= 0:
                    start = base + index
                    try:
                        info = _parse_godot_pck(file, start)
                    except (OSError, ValueError, UnicodeError):
                        file.seek(chunk_end)
                        search = index + 1
                        continue
                    return start, info
                file.seek(chunk_end)
                carry = haystack[-3:]
                position += len(chunk)
    except OSError:
        return None


def _godot_pck_candidates(root: pathlib.Path, executable: pathlib.Path) -> list[pathlib.Path]:
    names = {"data.pck"}
    if executable.is_file():
        names.update({executable.stem.lower() + ".pck", executable.name.lower() + ".pck"})
    try:
        candidates = [node for node in root.iterdir()
                      if node.is_file() and node.suffix.lower() == ".pck"]
        return [node for node in candidates
                if executable.is_dir() or node.name.lower() in names]
    except OSError:
        return []


def _read_pck_project_version(pack: pathlib.Path, start: int = 0) -> tuple[str | None, str | None, float, str | None]:
    try:
        with pack.open("rb") as file:
            info = _parse_godot_pck(file, start)
            if info.encrypted_directory:
                return None, "Godot PCK", 0.95, "Encrypted or unsupported Godot PCK"
            entry_name = "project.godot" if "project.godot" in info.entries else "project.binary"
            entry = info.entries.get(entry_name)
            if entry is None:
                return None, "Godot PCK", 0.95, "project.godot not found in Godot PCK"
            if entry.encrypted:
                return None, "Godot PCK", 0.95, "Encrypted or unsupported Godot PCK"
            if entry.size > _MAX_PROJECT_SETTINGS:
                return None, "Godot PCK", 0.95, "project.godot is unexpectedly large"
            file.seek(entry.offset)
            data = file.read(entry.size)
            if entry_name == "project.godot":
                version = _read_project_version(data)
                setting = "application/config/version"
                if version is None:
                    version = _read_project_version(data, "config/game_version")
                    setting = "application/config/game_version"
            else:
                version = _read_project_binary_setting(data)
                setting = "application/config/version"
                if version is None:
                    version = _read_project_binary_setting(data, "application/config/game_version")
                    setting = "application/config/game_version"
            source = f"PCK/{entry_name} {setting}"
            if version is None:
                fallback = _scan_godot_resource_versions(pack, start)
                if fallback is not None:
                    return fallback[0], fallback[1], 0.45, None
            return version, source, 1.0, None if version else "version not configured"
    except (OSError, ValueError, UnicodeError) as exc:
        return None, "Godot PCK", 0.8, f"Encrypted or unsupported Godot PCK: {exc}"


def _scan_godot_resource_versions(pack: pathlib.Path, start: int = 0) -> tuple[str, str] | None:
    """Find an explicit game-version field in small text resources as a fallback."""
    text_suffixes = {".gd", ".tscn", ".tres", ".cfg", ".ini", ".txt", ".json"}
    try:
        with pack.open("rb") as file:
            info = _parse_godot_pck(file, start)
            if info.encrypted_directory:
                return None
            scanned = 0
            candidates = []
            for path, entry in info.entries.items():
                if entry.encrypted or pathlib.PurePosixPath(path).suffix.lower() not in text_suffixes:
                    continue
                if entry.size > _MAX_GODOT_TEXT_RESOURCE or scanned + entry.size > _MAX_GODOT_RESOURCE_SCAN:
                    continue
                file.seek(entry.offset)
                data = file.read(entry.size)
                scanned += entry.size
                for pattern, priority in _GODOT_RESOURCE_VERSION_PATTERNS:
                    match = pattern.search(data)
                    if match:
                        version = match.group("version").decode("utf-8", errors="strict").strip()
                        candidates.append((priority, path, version))
                        break
            if not candidates:
                return None
            _, path, version = max(candidates, key=lambda item: (item[0], item[1].count("/"), item[1]))
            return version, f"PCK/{path} explicit version field (fallback)"
    except (OSError, ValueError, UnicodeError):
        return None


def _godot_signature(root: pathlib.Path, executable: pathlib.Path) -> tuple[bool, pathlib.Path | None, int | None, str | None]:
    """Return whether root has a validated Godot project pack or project file."""
    try:
        loose_project = root / "project.godot"
        if loose_project.is_file():
            return True, None, None, "loose project.godot"
    except OSError:
        pass
    for pack in _godot_pck_candidates(root, executable):
        try:
            with pack.open("rb") as file:
                info = _parse_godot_pck(file, 0)
            return True, pack, 0, "Godot PCK"
        except (OSError, ValueError, UnicodeError):
            continue
    if executable.is_file():
        embedded = _find_embedded_pck(executable)
        if embedded is not None:
            return True, executable, embedded[0], "embedded Godot PCK"
    return False, None, None, None


def detect_engine(executable: pathlib.Path | str) -> GameEngine:
    """Identify the installation engine independently from version extraction."""
    try:
        path = pathlib.Path(executable)
        root = _installation_root(path)
        if root is None:
            return GameEngine.Unknown
        if _unity_data_dir(path) is not None:
            return GameEngine.Unity
        identified, _, _, _ = _godot_signature(root, path)
        return GameEngine.Godot if identified else GameEngine.Unknown
    except (OSError, ValueError, TypeError):
        return GameEngine.Unknown


def detect_godot_version(executable: pathlib.Path | str) -> LocalVersionResult:
    """Extract application/config/version from loose or packed Godot settings."""
    path = pathlib.Path(executable)
    root = _installation_root(path)
    if root is None:
        return LocalVersionResult(None, GameEngine.Godot, None, error="Installation root not found")
    try:
        loose_project = root / "project.godot"
        if loose_project.is_file():
            if loose_project.stat().st_size > _MAX_PROJECT_SETTINGS:
                return LocalVersionResult(None, GameEngine.Godot, "project.godot", error="project.godot is unexpectedly large")
            version = _read_project_version(loose_project.read_bytes())
            source = "project.godot:application/config/version"
            if version is None:
                version = _read_project_version(loose_project.read_bytes(), "config/game_version")
                source = "project.godot:application/config/game_version"
            return LocalVersionResult(version, GameEngine.Godot, source, 1.0,
                                      None if version else "version not configured")
        loose_binary = root / "project.binary"
        if loose_binary.is_file():
            if loose_binary.stat().st_size > _MAX_PROJECT_SETTINGS:
                return LocalVersionResult(None, GameEngine.Godot, "project.binary", error="project.binary is unexpectedly large")
            binary = loose_binary.read_bytes()
            version = _read_project_binary_setting(binary)
            source = "project.binary:application/config/version"
            if version is None:
                version = _read_project_binary_setting(binary, "application/config/game_version")
                source = "project.binary:application/config/game_version"
            return LocalVersionResult(version, GameEngine.Godot, source, 1.0,
                                      None if version else "version not configured")
        for pack in _godot_pck_candidates(root, path):
            result = _read_pck_project_version(pack)
            if result[1] is not None:
                return LocalVersionResult(result[0], GameEngine.Godot, result[1], result[2], result[3])
        if path.is_file():
            embedded = _find_embedded_pck(path)
            if embedded is not None:
                result = _read_pck_project_version(path, embedded[0])
                return LocalVersionResult(result[0], GameEngine.Godot, result[1], result[2], result[3])
        return LocalVersionResult(None, GameEngine.Godot, "Godot PCK", 0.95,
                                  "version not configured or pack unsupported")
    except (OSError, ValueError, TypeError, UnicodeError) as exc:
        return LocalVersionResult(None, GameEngine.Godot, "Godot PCK", 0.8, str(exc) or exc.__class__.__name__)


def _extract_bundle_version(global_managers: pathlib.Path) -> tuple[str | None, str | None, float]:
    """Read PlayerSettings.bundleVersion using UnityPy.

    Unity serialized files are versioned and are not safely parseable with a
    regex. UnityPy is imported only when a Unity installation is identified,
    keeping the rest of the application usable if the optional parser is not
    installed.
    """
    try:
        import UnityPy

        # Loading bytes avoids UnityPy retaining an open Windows file handle.
        environment = UnityPy.load(global_managers.read_bytes())
        player_version = None
        player_raw_version = None
        player_object = None
        for obj in environment.objects:
            if getattr(obj.type, "name", None) != "PlayerSettings":
                continue
            player_object = obj
            try:
                data = obj.parse_as_object()
                version = getattr(data, "bundleVersion", None)
            except Exception:
                # Some Unity versions do not have a complete generated class;
                # the typetree dictionary can still expose this exact field.
                try:
                    data = obj.parse_as_dict()
                    version = data.get("bundleVersion") if isinstance(data, dict) else None
                except Exception:
                    version = None
            if isinstance(version, str) and version.strip():
                player_version = version.strip()
                break
            if version is None:
                try:
                    player_raw_version = _extract_raw_bundle_version(obj)
                except Exception:
                    player_raw_version = None
                break
        # Unity projects frequently leave the default PlayerSettings value at
        # 1.0 while a custom MonoBehaviour stores the actual release version.
        if player_version == "1.0" and player_object is not None:
            player_raw_version = _extract_raw_bundle_version(player_object)
        if player_version and player_version != "1.0":
            return player_version, "PlayerSettings.bundleVersion", 1.0
        if player_raw_version and player_raw_version != "1.0":
            return player_raw_version, "PlayerSettings.bundleVersion (raw fallback)", 0.4
        monobehaviour_version = _extract_raw_monobehaviour_version(environment)
        if monobehaviour_version:
            return monobehaviour_version, "MonoBehaviour raw version field", 0.25
        if player_version:
            return player_version, "PlayerSettings.bundleVersion", 1.0
        if player_raw_version:
            return player_raw_version, "PlayerSettings.bundleVersion (raw fallback)", 0.4
    except Exception as exc:
        raise RuntimeError(str(exc) or exc.__class__.__name__) from exc
    return None, None, 0.0


def detect_unity_version(executable: pathlib.Path | str) -> LocalVersionResult:
    """Extract a Unity version after the shared engine stage identified Unity."""
    try:
        path = pathlib.Path(executable)
        data_dir = _unity_data_dir(path)
        if data_dir is None:
            return LocalVersionResult(None, GameEngine.Unity, None, error="Unity data directory not found")

        unity_files = [data_dir / "globalgamemanagers", data_dir / "data.unity3d"]
        unity_files = [path for path in unity_files if path.is_file()]
        if not unity_files:
            return LocalVersionResult(
                None, GameEngine.Unity, None, error="Unity serialized data file not found"
            )

        parse_error = None
        for unity_file in unity_files:
            try:
                version, source, confidence = _extract_bundle_version(unity_file)
            except Exception as exc:
                parse_error = exc
                continue
            if version is not None:
                break
        else:
            return LocalVersionResult(
                None, GameEngine.Unity, None,
                error=f"Unable to parse Unity data: {parse_error}" if parse_error else "PlayerSettings.bundleVersion not found"
            )
        if version is None:
            return LocalVersionResult(
                None, GameEngine.Unity, None, error="PlayerSettings.bundleVersion not found"
            )
        return LocalVersionResult(
            version, GameEngine.Unity, source, confidence=confidence,
        )
    except (OSError, ValueError, TypeError) as exc:
        return LocalVersionResult(None, GameEngine.Unity, None, error=str(exc) or exc.__class__.__name__)


def detect_installed_version(executable: pathlib.Path | str) -> LocalVersionResult:
    """Detect a local version through engine detection and dispatch."""
    engine = detect_engine(executable)
    if engine is GameEngine.Unity:
        return detect_unity_version(executable)
    if engine is GameEngine.Godot:
        return detect_godot_version(executable)
    return LocalVersionResult(None, None, None, error="Unknown game engine")


def detect_game_installed_version(game: typing.Any) -> LocalVersionResult:
    """Try the first valid local executable linked to ``game``."""
    for result in detect_game_installed_versions(game):
        if result.engine in (GameEngine.Unity, "Unity", GameEngine.Godot, "Godot") or result.version is not None:
            return result
    return LocalVersionResult(None, None, None, error="No linked executable found")


def detect_game_installed_versions(game: typing.Any) -> list[LocalVersionResult]:
    """Detect each linked executable independently, preserving array order."""
    from modules import globals, utils

    results = []
    for executable in game.executables:
        if utils.is_uri(executable):
            results.append(LocalVersionResult(None, None, None, error="URI is not a local file"))
            continue
        path = pathlib.Path(executable)
        base = globals.settings.default_exe_dir.get(globals.os)
        if not path.is_absolute() and base:
            path = pathlib.Path(base) / path
        results.append(detect_installed_version(path))
    return results
