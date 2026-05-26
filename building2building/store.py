import hashlib
import inspect
import logging
import os
import pickle
import shutil
import tarfile
import tempfile
import zipfile
from contextvars import ContextVar
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar, Union, overload
from uuid import uuid4

import git
import requests
from rich.progress import (
    DownloadColumn,
    FileSizeColumn,
    Progress,
    ProgressColumn,
    TransferSpeedColumn,
)

logger = logging.getLogger(__name__)

# For displaying progress bars
PROGRESS: ContextVar[Progress] = ContextVar("PROGRESS")

# Context variable for the current derivation output path
OUTPUT: ContextVar[Path] = ContextVar("OUTPUT")


@dataclass(frozen=True)
class Derivation:
    name: str
    hash: bytes
    dependencies: list["Realizable"]
    # Builder now receives only realized dependencies; output path is read from the ContextVar
    builder: Callable[[list[Any]], None]

    def __post_init__(self):
        assert (
            Path(self.name).name == self.name
        ), "name of derivation can't contain a slash"


Result = TypeVar("Result")


@dataclass(frozen=True)
class Expression(Generic[Result]):
    hash: bytes
    dependencies: list["Realizable"]
    builder: Callable[[list[Any]], Result]


Realizable = Union[Derivation, Expression]


@overload
def realize(store_path: Path, expression: Expression[Result]) -> Result: ...
@overload
def realize(store_path: Path, derivation: Derivation) -> Path: ...
def realize(store_path: Path, realizable: Realizable) -> Path | Result:
    """Realize a derivation (returns Path) or expression (returns value)."""

    def inner(realizable: Realizable):
        if isinstance(realizable, Derivation):
            # Check if already built
            output_path = store_path / (realizable.hash.hex() + "-" + realizable.name)
            if output_path.exists():
                return output_path

            # Build into a unique temporary path, then atomically rename into place.
            # This prevents:
            # - partially-built outputs from being treated as complete
            # - concurrent processes from clobbering the same output directory/files
            tmp_output_path = store_path / (
                f"{realizable.hash.hex()}-{realizable.name}.tmp-{os.getpid()}-{uuid4().hex}"
            )

            # Realize dependencies
            realized_deps = [inner(dep) for dep in realizable.dependencies]

            # Build with ContextVar carrying the temp output path
            tmp_output_path.parent.mkdir(parents=True, exist_ok=True)
            token_output = OUTPUT.set(tmp_output_path)
            try:
                logger.info(f"building {realizable.name}")
                realizable.builder(realized_deps)

                if not tmp_output_path.exists():
                    file_path = inspect.getsourcefile(realizable.builder)
                    source_lines, start_line = inspect.getsourcelines(
                        realizable.builder
                    )

                    raise Exception(
                        f"derivation {realizable.name} didn't product an output. Perhaps make the builder at {file_path}:{start_line} not silently fail?"
                    )
                # If another process finished first, keep the existing output and
                # clean up our temp output.
                if output_path.exists():
                    try:
                        if tmp_output_path.is_dir():
                            shutil.rmtree(tmp_output_path, ignore_errors=True)
                        else:
                            tmp_output_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return output_path

                try:
                    tmp_output_path.rename(output_path)
                except FileExistsError:
                    # Race: another process created it between our check and rename.
                    try:
                        if tmp_output_path.is_dir():
                            shutil.rmtree(tmp_output_path, ignore_errors=True)
                        else:
                            tmp_output_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    return output_path
            finally:
                OUTPUT.reset(token_output)
            return output_path

        elif isinstance(realizable, Expression):
            # Realize dependencies and evaluate (no caching)
            realized_deps = [inner(dep) for dep in realizable.dependencies]
            result = realizable.builder(realized_deps)
            return result

        else:
            raise ValueError(f"Unknown realizable type: {type(realizable)}")

    return inner(realizable)


def compute_hash(
    func, name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> bytes:
    """Compute a content hash for an expression/derivation input.

    - Includes the function name.
    - For Realizable args/kwargs, uses their .hash directly.
    - For everything else, uses pickle to serialize and hash the bytes.
    """
    hasher = hashlib.blake2b(digest_size=32)

    if inspect.isfunction(func):
        hasher.update(func.__name__.encode("utf-8"))
        hasher.update(inspect.getsource(func).encode("utf-8"))

    hasher.update(name.encode("utf-8"))

    # Positional args
    for arg in args:
        if isinstance(arg, (Derivation, Expression)):
            hasher.update(arg.hash)
        else:
            payload = pickle.dumps(arg)
            hasher.update(payload)

    # Keyword args (preserve call order)
    for key, value in kwargs.items():
        hasher.update(str(key).encode("utf-8"))
        if isinstance(value, (Derivation, Expression)):
            hasher.update(value.hash)
        else:
            payload = pickle.dumps(value)
            hasher.update(payload)

    return hasher.digest()


def _capture_dependencies_and_builder(func: Callable[..., Any], *args, **kwargs):
    """Internal helper used by the decorators to:
    - capture Realizable dependencies in call order
    - build placeholder structures to reconstruct args/kwargs
    - return (dependencies, builder)
    The resulting builder takes a single argument: the list of realized dependencies
    in the same order they were captured.
    """
    dependencies: list[Realizable] = []

    # Placeholders for reconstruction
    pos_placeholders: list[Any | None] = []
    for a in args:
        if isinstance(a, (Derivation, Expression)):
            dependencies.append(a)
            pos_placeholders.append(None)
        else:
            pos_placeholders.append(a)

    kw_placeholders: dict[str, Any | None] = {}
    for k, v in kwargs.items():
        if isinstance(v, (Derivation, Expression)):
            dependencies.append(v)
            kw_placeholders[k] = None
        else:
            kw_placeholders[k] = v

    @wraps(func)
    def builder(realized_deps: list[Any]) -> Any:
        dep_iter = iter(realized_deps)

        final_args: list[Any] = []
        for val in pos_placeholders:
            if val is None:
                final_args.append(next(dep_iter))
            else:
                final_args.append(val)

        final_kwargs: dict[str, Any] = {}
        for k, v in kw_placeholders.items():
            if v is None:
                final_kwargs[k] = next(dep_iter)
            else:
                final_kwargs[k] = v

        return func(*final_args, **final_kwargs)

    return dependencies, builder


def expression() -> (
    Callable[[Callable[..., Result]], Callable[..., Expression[Result]]]
):
    """Decorator: calling the wrapped function returns an Expression node."""

    def decorator(func) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            dependencies, builder = _capture_dependencies_and_builder(
                func, *args, **kwargs
            )
            return Expression(
                hash=compute_hash(func, "expr", args, kwargs),
                dependencies=dependencies,
                builder=builder,
            )

        return wrapper

    return decorator


def derivation(
    name: str | Callable,
) -> Callable[[Callable[..., None]], Callable[..., Derivation]]:
    """Decorator: calling the wrapped function returns a Derivation node.

    The constructed builder takes only the realized dependency values. During
    realization, the output path is made available via the ContextVar
    `current_output_path`.
    """

    def compute_name(args, kwargs):
        if isinstance(name, str):
            return name
        elif isinstance(name, Callable):
            return name(*args, **kwargs)

    def decorator(func) -> Callable[..., Derivation]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Derivation:
            dependencies, builder = _capture_dependencies_and_builder(
                func, *args, **kwargs
            )

            der_name = compute_name(args, kwargs)
            return Derivation(
                name=der_name,
                hash=compute_hash(func, der_name, args, kwargs),
                dependencies=dependencies,
                # builder returns None; output path is accessible via current_output_path
                builder=builder,
            )

        return wrapper

    return decorator


# concrete implementations


def DownloadFile(
    filename: str,
    url: str,
    hash: bytes | None,
    hasher_factory=hashlib.sha256,
) -> Derivation:
    if hash is not None:
        derivation_hash = hash
    else:
        h = hashlib.sha256()
        h.update(url.encode("utf-8"))
        derivation_hash = h.digest()

    def builder(_):
        out = OUTPUT.get()

        # Get progress object and create a task

        with Progress(
            *Progress.get_default_columns(),
            DownloadColumn(),
            TransferSpeedColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(f"Downloading {filename}", total=None)

            hasher = hasher_factory()
            # Verify TLS certificates by default.
            #
            # If your environment requires a custom CA bundle, set one of:
            # - B2B_CA_BUNDLE=/path/to/ca-bundle.pem
            # - REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
            # - SSL_CERT_FILE=/path/to/ca-bundle.pem
            #
            # To explicitly opt-out (NOT recommended), set:
            # - B2B_INSECURE_SSL=1
            verify: bool | str = True
            if os.environ.get("B2B_INSECURE_SSL", "").strip().lower() in (
                "1",
                "true",
                "yes",
            ):
                verify = False
            else:
                ca_bundle = (
                    os.environ.get("B2B_CA_BUNDLE")
                    or os.environ.get("REQUESTS_CA_BUNDLE")
                    or os.environ.get("SSL_CERT_FILE")
                )
                if ca_bundle:
                    verify = ca_bundle

            response = requests.get(url, stream=True, verify=verify)
            response.raise_for_status()

            # Update task with actual file size if available
            total_size = int(response.headers.get("content-length", 0))
            if total_size > 0:
                progress.update(task, total=total_size)

            block_size = 1 << 16

            with tempfile.NamedTemporaryFile("wb", delete=False) as outfile:
                for data in response.iter_content(block_size):
                    hasher.update(data)
                    outfile.write(data)
                    progress.update(task, advance=len(data))

            outfile.close()
            h = hasher.digest()
            if hash is not None and hash != h:
                raise Exception(
                    f"Hash of download {filename} is wrong. Expected: {hash.hex()}, actual: {h.hex()} (computed using {hasher})"
                )
            shutil.move(outfile.name, out)

    return Derivation(filename, derivation_hash, [], builder)


def ExtractTarball(input_der: Derivation):
    @derivation(input_der.name.removesuffix(".tar.gz"))
    def inner(input: Path):
        output = OUTPUT.get()

        with tarfile.open(input, "r:gz") as tar:
            tar.extractall(path=output)

    return inner(input_der)


def ExtractZip(input_der: Derivation):
    @derivation(input_der.name.removesuffix(".zip"))
    def inner(input: Path):
        dst = OUTPUT.get()

        dst.mkdir()

        with zipfile.ZipFile(input, "r") as zip_ref:
            file_list = zip_ref.infolist()

            for file_info in file_list:
                zip_ref.extract(file_info, dst)

    return inner(input_der)


def ExtractFromZip(zip_file: Realizable, filename: str) -> Derivation:
    @derivation(Path(filename).name)
    def inner(input: Path, name: str):
        dst = OUTPUT.get()
        with zipfile.ZipFile(input, "r") as zip_ref:
            with zip_ref.open(filename) as src, open(dst, "wb") as dst:
                shutil.copyfileobj(src, dst)

    return inner(zip_file, filename)


def hash_directory_tree(hasher, dir: Path):
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


def GitClone(
    filename: str,
    url: str,
    commit: str,
    expected_hash: bytes,
    hasher_factory=hashlib.sha256,
) -> Derivation:
    def builder(_):
        dst = OUTPUT.get()

        hasher = hasher_factory()

        with tempfile.TemporaryDirectory() as tempdir:
            tempdir_path = Path(tempdir)

        repo = git.Repo.clone_from(url, tempdir_path)
        correct_commit = repo.create_head("correct_commit", commit)
        repo.head.reference = correct_commit
        assert not repo.head.is_detached
        # Reset the index and working tree to match the pointed-to commit.
        repo.head.reset(index=True, working_tree=True)

        repo.close()
        shutil.rmtree(tempdir_path / ".git")

        hash_directory_tree(hasher, tempdir_path)
        h = hasher.digest()
        if h != expected_hash:
            raise Exception(
                f"Hash of git repo {filename} is wrong. Expected: {expected_hash.hex()}, actual: {h.hex()} (computed using {hasher})"
            )

        shutil.move(tempdir_path, dst)

    return Derivation(filename, expected_hash, [], builder)


def LocalFile(filepath: Path) -> Derivation:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    h = hasher.digest()

    def builder(_):
        dst = OUTPUT.get()

        shutil.copy(filepath, dst)

    return Derivation(filepath.name, h, [], builder)


def LocalSymlink(name: str, filepath: Path) -> Derivation:
    def builder(_):
        dst = OUTPUT.get()

        dst.symlink_to(filepath)

    hasher = hashlib.sha256()
    hasher.update(str(filepath).encode("utf-8"))

    return Derivation(name, hasher.digest(), [], builder)


FileLike = Union[Derivation, Expression[Path]]


def Symlink(name: str, input: FileLike) -> Derivation:
    @derivation(name)
    def builder(input):
        dst = OUTPUT.get()
        dst.symlink_to(dst)

    return builder(input)


def Rename(name: str, input: Realizable) -> Derivation:
    @derivation(name)
    def builder(input: Path):
        dst = OUTPUT.get()
        if input.is_file():
            shutil.copy(input, dst)
        elif input.is_dir():
            shutil.copytree(input, dst)
        else:
            raise Exception(f"don't know what to do with a file like {input}")

    return builder(input)


@expression()
def ChildFile(parent: Path, child: str) -> Path:
    return parent / child


@expression()
def Constant(x):
    return x
