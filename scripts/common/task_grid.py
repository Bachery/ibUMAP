from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Iterable, Mapping, Sequence


@dataclass(frozen=True)
class EmbeddingTask:
    dataset: str
    variant: str
    seed: int | None = None
    repeat: int | None = None

    @property
    def run_id(self) -> str:
        parts = [self.dataset, self.variant]
        if self.repeat is not None:
            parts.append(f"repeat_{int(self.repeat)}")
        if self.seed is not None:
            parts.append(f"seed_{int(self.seed)}")
        return "__".join(parts)


def entry_name(entry: Any, *, keys: Sequence[str] = ("name", "dataset_id", "id", "path")) -> str:
    if isinstance(entry, str):
        return entry
    if isinstance(entry, Mapping):
        for key in keys:
            value = entry.get(key)
            if value is not None:
                return str(value)
    raise ValueError(f"Entry must be a string or define one of {tuple(keys)}: {entry!r}")


def selected_names(
    requested: Sequence[str] | None,
    available: Iterable[str],
    *,
    kind: str,
) -> tuple[str, ...]:
    available_names = tuple(str(value) for value in available)
    if not requested:
        return available_names
    requested_names = tuple(str(value) for value in requested)
    unknown = sorted(set(requested_names) - set(available_names))
    if unknown:
        raise ValueError(f"Unknown {kind}(s): {', '.join(unknown)}")
    return requested_names


def dataset_entries(config: Mapping[str, Any]) -> list[Any]:
    return list(config.get("datasets") or [])


def algorithm_entries(config: Mapping[str, Any]) -> list[Any]:
    return list(config.get("algorithms") or config.get("variants") or [])


def filter_entries(entries: Sequence[Any], filters: Sequence[str] | None, *, kind: str) -> list[Any]:
    if not filters:
        return list(entries)
    wanted = set(str(value) for value in filters)
    output = [entry for entry in entries if entry_name(entry) in wanted]
    found = {entry_name(entry) for entry in output}
    unknown = sorted(wanted - found)
    if unknown:
        raise ValueError(f"Unknown {kind}(s): {', '.join(unknown)}")
    return output


def cartesian_embedding_tasks(
    datasets: Sequence[str],
    variants: Sequence[str],
    *,
    seeds: Sequence[int | None] = (None,),
    repeats: Sequence[int | None] = (None,),
) -> list[EmbeddingTask]:
    return [
        EmbeddingTask(dataset=dataset, variant=variant, seed=seed, repeat=repeat)
        for dataset, variant, seed, repeat in product(datasets, variants, seeds, repeats)
    ]


def group_tasks_by_dataset(tasks: Iterable[EmbeddingTask]) -> dict[str, list[EmbeddingTask]]:
    grouped: dict[str, list[EmbeddingTask]] = {}
    for task in tasks:
        grouped.setdefault(task.dataset, []).append(task)
    return grouped


def print_embedding_dry_run(tasks: Sequence[EmbeddingTask], *, header: str = "Dry run only") -> None:
    print(header)
    print(f"Task count: {len(tasks)}")
    for task in tasks:
        parts = [f"dataset={task.dataset}", f"variant={task.variant}"]
        if task.repeat is not None:
            parts.append(f"repeat={task.repeat}")
        if task.seed is not None:
            parts.append(f"seed={task.seed}")
        print("- " + " ".join(parts))

