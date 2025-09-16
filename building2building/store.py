# This is a tiny set of utilities for maintaining a content addressed store of
# artifaces. Inspired by Nix.
import contextlib
import hashlib
import logging
import os
import shutil
import tarfile
import tempfile
import zipfile
from abc import abstractmethod
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Protocol, runtime_checkable

import git
import requests
from rich import console, progress
from rich.live import Live
from rich.tree import Tree

logger = logging.getLogger(__name__)


@contextmanager
def set_contextvar(var: ContextVar, value):
    token = var.set(value)
    try:
        yield
    finally:
        var.reset(token)


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


current_progress_tree: ContextVar[Tree] = ContextVar("current_progress_tree")


def build(store_path: Path, step: Derivation) -> Path:
    def inner(step: Derivation, live: Live) -> Path:
        progress_tree = current_progress_tree.get()

        node = progress_tree.add(
            f"[bold]{step.name()}[/bold] [yellow](building dependencies...)[/yellow]"
        )

        def over():
            node.label = f"[bold]{output_name}[/bold] [green]✓[/green]"

        with contextlib.ExitStack() as stack:
            stack.callback(over)

            with set_contextvar(current_progress_tree, node):
                output_name = step.name()
                h = step.hash()
                loc = store_path / (h.hex() + "-" + output_name)

                # Add this step to the tree
                if loc.exists():
                    return loc

                loc.parent.mkdir(exist_ok=True)

                deps = {}
                for name, dep in step.dependencies().items():
                    deps[name] = inner(dep, live)

                # Update status to show we're building this step
                node.label = f"[bold]{output_name}[/bold] [blue](building...)[/blue]"
                step.build(loc, deps)

                # Mark as complete

        return loc

    tree = Tree("Build Process")
    with Live(tree, refresh_per_second=10) as live:
        with set_contextvar(current_progress_tree, tree):
            result = inner(step, live)

    return result


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
        progress_tree = current_progress_tree.get()
        # url: str, dest: Path, description: str | None = None, verify: bool = False

        # We don't need to verify, we will be checking the hash
        response = requests.get(self.url, stream=True, verify=False)

        response.raise_for_status()

        block_size = 1 << 16

        hasher = self.hasher

        # Get total file size from headers
        total_size = int(response.headers.get("content-length", 0))

        with tempfile.NamedTemporaryFile("wb", delete=False) as outfile:
            progress_bar = progress.Progress(
                "[progress.description]{task.description}",
                progress.BarColumn(),
                progress.TaskProgressColumn(),
                progress.DownloadColumn(),
                progress.TransferSpeedColumn(),
                progress.TimeRemainingColumn(),
                console=None,
            )
            task = progress_bar.add_task(f"Downloading {self.name()}", total=total_size)
            for data in response.iter_content(block_size):
                progress_bar.update(task, advance=len(data))
                progress_tree.label = progress_bar.get_renderable()
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

            progress_bar = progress.Progress(
                "[progress.description]{task.description}",
                progress.BarColumn(),
                progress.TaskProgressColumn(),
                progress.DownloadColumn(),
                progress.TransferSpeedColumn(),
                progress.TimeRemainingColumn(),
            )
            task = progress_bar.add_task(f"Extracting {self.name()}", total=total_size)
            progress_tree = current_progress_tree.get()
            for file_info in file_list:
                zip_ref.extract(file_info, dst)
                progress_bar.update(task, advance=file_info.file_size)
                progress_tree.label = progress_bar.get_renderable()


@dataclass
class ExtractTarball(BaseDerivation):
    input: Derivation

    def name(self) -> str:
        return self.input.name().removesuffix(".tar.gz")

    def build(self, dst: Path, deps):
        # Extract to specific directory
        with tarfile.open(deps["input"], "r:gz") as tar:
            tar.extractall(path=dst)


@dataclass
class Child(BaseDerivation):
    input: Derivation
    path: str

    def name(self) -> str:
        return Path(self.path).name

    def hash(self) -> bytes:
        hasher = hashlib.sha256()
        hasher.update(self.salt())
        hasher.update(self.input.hash())
        hasher.update(self.path.encode("utf-8"))
        return hasher.digest()

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

    _name: str

    destination: Path

    def __init__(self, _name: str, destination: Path) -> None:
        self._name = _name
        self.destination = destination.resolve()

    def name(self) -> str:
        return self._name

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


def hash_directory_tree(hasher: Hasher, dir: Path):
    # Get all files and sort them for deterministic ordering
    file_paths = []
    for root, dirs, files in os.walk(dir):
        # Sort directories and files for consistent ordering
        dirs.sort()
        files.sort()
        for file in files:
            file_paths.append(os.path.join(root, file))

    # Sort all file paths to ensure deterministic order
    file_paths.sort()

    # Hash each file's content
    for file_path in file_paths:
        # Include the relative path in the hash for structure integrity
        rel_path = os.path.relpath(file_path, dir)
        hasher.update(rel_path.encode("utf-8"))

        # Hash the file content
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                hasher.update(chunk)


class GitRemoteProgress(git.RemoteProgress):
    """Stolen from https://stackoverflow.com/a/71285627"""

    OP_CODES = [
        "BEGIN",
        "CHECKING_OUT",
        "COMPRESSING",
        "COUNTING",
        "END",
        "FINDING_SOURCES",
        "RECEIVING",
        "RESOLVING",
        "WRITING",
    ]
    OP_CODE_MAP = {
        getattr(git.RemoteProgress, _op_code): _op_code for _op_code in OP_CODES
    }

    progressbar: progress.Progress
    tree: Tree

    def __init__(self, tree: Tree) -> None:
        super().__init__()
        self.tree = tree
        self.progressbar = progress.Progress(
            progress.SpinnerColumn(),
            # *progress.Progress.get_default_columns(),
            progress.TextColumn("[progress.description]{task.description}"),
            progress.BarColumn(),
            progress.TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            "eta",
            progress.TimeRemainingColumn(),
            progress.TextColumn("{task.fields[message]}"),
            transient=False,
        )
        self.active_task = None

    def __del__(self) -> None:
        # logger.info("Destroying bar...")
        # self.progressbar.stop()
        pass

    @classmethod
    def get_curr_op(cls, op_code: int) -> str:
        """Get OP name from OP code."""
        # Remove BEGIN- and END-flag and get op name
        op_code_masked = op_code & cls.OP_MASK
        return cls.OP_CODE_MAP.get(op_code_masked, "?").title()

    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str | None = "",
    ) -> None:
        # Start new bar on each BEGIN-flag
        if op_code & self.BEGIN:
            self.curr_op = self.get_curr_op(op_code)
            # logger.info("Next: %s", self.curr_op)
            self.active_task = self.progressbar.add_task(
                description=self.curr_op,
                total=max_count,
                message=message,
            )

        self.progressbar.update(
            task_id=self.active_task,
            completed=cur_count,
            message=message,
        )
        self.tree.label = self.progressbar.get_renderable()

        # End progress monitoring on each END-flag
        if op_code & self.END:
            # logger.info("Done: %s", self.curr_op)
            self.progressbar.update(
                task_id=self.active_task,
                message=f"[bright_black]{message}",
            )


@dataclass
class GitClone(Derivation):
    filename: str
    url: str
    commit: str
    expected_hash: bytes
    hasher: Hasher = field(default_factory=hashlib.sha256)

    def dependencies(self) -> dict[str, Derivation]:
        return {}

    def name(self) -> str:
        return self.filename

    def salt(self) -> bytes:
        return "git-download".encode("utf-8")

    def hash(self) -> bytes:
        return self.expected_hash

    def build(self, dst: Path, *_):
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)

            progress_tree = current_progress_tree.get()
            repo = git.Repo.clone_from(
                self.url, tempdir_path, progress=GitRemoteProgress(progress_tree)
            )
            correct_commit = repo.create_head("correct_commit", self.commit)
            repo.head.reference = correct_commit
            assert not repo.head.is_detached
            # Reset the index and working tree to match the pointed-to commit.
            repo.head.reset(index=True, working_tree=True)

            repo.close()

            shutil.rmtree(tempdir_path / ".git")

            hash_directory_tree(self.hasher, tempdir_path)
            h = self.hasher.digest()
            if h != self.expected_hash:
                raise Exception(
                    f"Hash of git repo {self.name()} is wrong. Expected: {self.expected_hash.hex()}, actual: {h.hex()} (computed using {self.hasher})"
                )

            shutil.move(tempdir_path, dst)
