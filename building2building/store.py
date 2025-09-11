# This is a tiny set of utilities for maintaining a content addressed store of
# artifaces. Inspired by Nix.
import hashlib
import logging
import shutil
import tarfile
import tempfile
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)


@runtime_checkable
class Derivation(Protocol):
    """A derivation represent a resource (a file) whose realisation depends on
    other derivation's output. Together, many derivations form a directed
    acyclic graph. The outputs of derivations are stored in files whose names
    are a hash of all the dependencies, making them content addressable.

    """

    def name(self) -> str:
        """Human readable name.

        This name will be part of the output's file name in the store.

        """
        ...

    def salt(self) -> bytes:
        """Identifier for the build method used."""
        ...

    def dependencies(self) -> dict[str, "Derivation"]:
        """List the direct dependencies.

        When the derivation is realised, the dependencies' file paths will be
        given to the run method.

        """

        ...

    def build(
        self,
        dst: Path,
        deps: dict[str, Path],
        /,
    ):
        """Realise the derivation by writing to `dst` the results."""
        ...

    def hash(self) -> bytes:
        """Compute the content address of this derivation."""
        ...


class BaseDerivation:
    def hash(self) -> bytes:
        return default_hash(self.salt(), self.dependencies())

    def salt(self) -> bytes:
        """Auto-generate salt from class name."""
        return self.__class__.__name__.encode("utf-8")

    def dependencies(self) -> dict[str, Derivation]:
        """Auto-detect dependencies from dataclass fields."""
        deps = {}

        for f in fields(self):
            value = getattr(self, f.name)

            if f.type == Derivation:
                deps[f.name] = value

        return deps

    @abstractmethod
    def build(
        self,
        dst: Path,
        deps: dict[str, Path],
        /,
    ): ...


@runtime_checkable
class Hasher(Protocol):
    def update(self, x: bytes, /): ...
    def digest(self) -> bytes: ...


def default_hash(salt: bytes, deps: dict[str, Derivation]) -> bytes:
    """The default hash method.

    This hash method computes the hash using the salt identifying the builder
    and the dependencies. One would only need something else when creating new
    "leaf" derivation types.

    """
    hasher = hashlib.sha256()
    hasher.update(salt)
    for dep in deps.values():
        hasher.update(dep.hash())

    return hasher.digest()


@dataclass
class InputFile(Derivation):
    filepath: Path

    def __init__(self, filepath: Path) -> None:
        self.filepath = filepath

    def dependencies(self) -> dict[str, Derivation]:
        return {}

    def name(self) -> str:
        return self.filepath.name

    def salt(self) -> bytes:
        return "file-system".encode("utf-8")

    def hash(self) -> bytes:
        hasher = hashlib.sha256()
        with open(self.filepath, "rb") as f:
            hasher.update(f.read())
        return hasher.digest()

    def build(self, dst, *_):
        shutil.copy(self.filepath, dst)


@dataclass
class DownloadFile(Derivation):
    filename: str
    url: str
    expected_hash: bytes
    hasher: Hasher = field(default_factory=hashlib.sha256)

    def dependencies(self) -> dict[str, Derivation]:
        return {}

    def name(self) -> str:
        return self.filename

    def salt(self) -> bytes:
        return "url-download".encode("utf-8")

    def hash(self) -> bytes:
        return self.expected_hash

    def build(self, dst: Path, *_):
        # url: str, dest: Path, description: str | None = None, verify: bool = False

        # We don't need to verify, we will be checking the hash
        response = requests.get(self.url, stream=True, verify=False)

        response.raise_for_status()

        block_size = 1 << 16

        hasher = self.hasher

        # Get total file size from headers
        total_size = int(response.headers.get("content-length", 0))

        with tempfile.NamedTemporaryFile("wb", delete=False) as outfile:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {self.name()}",
            ) as progress_bar:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    hasher.update(data)
                    outfile.write(data)

        outfile.close()
        # We compute extract the hash and check it is the same as the one it was
        # supposed to be.

        h = hasher.digest()

        if self.expected_hash != h:
            raise Exception(
                f"Hash of download {self.name()} is wrong. Expected: {self.expected_hash.hex()}, actual: {h.hex()} (computed using {self.hasher})"
            )

        shutil.move(outfile.name, dst)


@dataclass
class ExtractZip(BaseDerivation):
    filename: str
    file: Derivation

    def name(self) -> str:
        return self.filename

    def build(
        self,
        dst: Path,
        deps,
    ):
        dst.mkdir()

        with zipfile.ZipFile(deps["file"], "r") as zip_ref:
            file_list = zip_ref.infolist()

            # Calculate total uncompressed size
            total_size = sum(file_info.file_size for file_info in file_list)

            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"extracting {self.name()}",
            ) as progress_bar:
                for file_info in file_list:
                    zip_ref.extract(file_info, dst)
                    progress_bar.update(file_info.file_size)


@dataclass
class ExtractTarball(BaseDerivation):
    input: Derivation

    def name(self) -> str:
        return self.input.name().removesuffix(".tar.gz")

    def build(self, dst: Path, *deps):
        (p,) = deps

        # Extract to specific directory
        with tarfile.open(p, "r:gz") as tar:
            tar.extractall(path=dst)


@dataclass
class Child(BaseDerivation):
    input: Derivation
    path: str

    def name(self) -> str:
        return Path(self.path).name

    def build(
        self,
        dst: Path,
        deps,
    ):
        dst.symlink_to(deps["input"] / self.path)


@dataclass
class Symlink(Derivation):
    """Computes a symlink pointing to an arbitrary place.

    Useful for pointing to files outside the store.

    """

    destination: Path

    def __init__(self, destination: Path) -> None:
        self.destination = destination.resolve()

    def name(self) -> str:
        return self.destination.name

    def dependencies(self) -> dict[str, Derivation]:
        return {}

    def hash(self) -> bytes:
        hasher = hashlib.sha256()
        hasher.update(str(self.destination).encode("utf-8"))

        return hasher.digest()

    def salt(self) -> bytes:
        return "symlink-child".encode("utf-8")

    def build(
        self,
        dst: Path,
        _,
    ):
        dst.symlink_to(self.destination)


def build(store_path: Path, step: Derivation) -> Path:
    name = step.name()
    h = step.hash()
    loc = store_path / (h.hex() + "-" + name)

    if loc.exists():
        return loc

    loc.parent.mkdir(exist_ok=True)

    deps = {}
    for name, dep in step.dependencies().items():
        deps[name] = build(store_path, dep)

    logger.info(f"Building {name}")
    step.build(loc, deps)

    return loc
