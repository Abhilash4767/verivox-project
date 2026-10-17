from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from .. import storage
from ..features import load_audio
from ..schemas import DatasetItem, DatasetStats
from ..security import require_api_key

router = APIRouter(prefix="/dataset", tags=["dataset"], dependencies=[Depends(require_api_key)])


_cached_stats: tuple[int, int, float] | None = None  # (real_count, fake_count, total_seconds)


@router.get("/stats", response_model=DatasetStats)
async def stats() -> DatasetStats:
    global _cached_stats
    real = storage.list_dataset("real")
    fake = storage.list_dataset("fake")
    total = len(real) + len(fake)
    total_seconds = 0.0

    if _cached_stats and _cached_stats[0] == len(real) and _cached_stats[1] == len(fake):
        total_seconds = _cached_stats[2]
    elif total > 0:
        sample_size = min(len(real) + len(fake), 20)
        import random
        all_files = real + fake
        sampled = random.sample(all_files, sample_size)
        sample_sec = 0.0
        sampled_count = 0
        for p in sampled:
            try:
                import wave
                with wave.open(str(p), "rb") as w:
                    sample_sec += w.getnframes() / float(w.getframerate())
                    sampled_count += 1
            except Exception:
                continue
        avg_sec = (sample_sec / sampled_count) if sampled_count > 0 else 3.5
        total_seconds = avg_sec * total
        _cached_stats = (len(real), len(fake), total_seconds)

    balance = (min(len(real), len(fake)) / max(1, max(len(real), len(fake)))) if total else 0.0
    return DatasetStats(
        real=len(real),
        fake=len(fake),
        total=total,
        total_seconds=round(total_seconds, 2),
        balance=round(balance, 4),
    )


@router.get("/items", response_model=list[DatasetItem])
async def items(label: str | None = None) -> list[DatasetItem]:
    labels = [label] if label in {"real", "fake"} else ["real", "fake"]
    out: list[DatasetItem] = []
    for lb in labels:
        for p in storage.list_dataset(lb):
            out.append(
                DatasetItem(
                    file_name=p.name, label=lb, size_kb=round(p.stat().st_size / 1024, 2)
                )
            )
    return out


@router.post("/upload", response_model=DatasetItem, status_code=201)
async def upload(label: str = Form(...), file: UploadFile = File(...)) -> DatasetItem:
    if label not in {"real", "fake"}:
        raise HTTPException(422, "label must be 'real' or 'fake'")
    content = await file.read()
    if not content:
        raise HTTPException(422, "Empty file")
    path = storage.save_dataset_file(label, file.filename or "clip.wav", content)
    return DatasetItem(
        file_name=path.name, label=label, size_kb=round(len(content) / 1024, 2)
    )


@router.delete("/items/{label}/{file_name}", status_code=204)
async def delete_item(label: str, file_name: str) -> None:
    if label not in {"real", "fake"}:
        raise HTTPException(422, "label must be 'real' or 'fake'")
    target = storage.dataset_dir(label) / storage.safe_name(file_name)
    if not target.exists():
        raise HTTPException(404, "Not found")
    target.unlink()
    storage.invalidate_dataset_cache(label)
