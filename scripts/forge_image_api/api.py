"""Public API for PARAM FORGE."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, List, Optional

from forge_image_api.core.contracts import ImageEvent, ImageInputs, ImageRequest, ImageResult
from forge_image_api.core.receipts import build_receipt, write_receipt
from forge_image_api.core.router import resolve_provider
from forge_image_api.core.solver import resolve_request
from forge_image_api.core.utils import ensure_out_dir, utc_timestamp
from forge_image_api.core.streaming import wrap_final_results
from forge_image_api.providers import get_adapter
from forge_image_api.providers.base import ProviderImage, ProviderResponse, ProviderStreamEvent


def _extension_from_mime(mime_type: Optional[str], fallback: str) -> str:
    if mime_type:
        mime = mime_type.lower()
        if mime.endswith("/jpeg") or mime.endswith("/jpg"):
            return "jpg"
        if mime.endswith("/png"):
            return "png"
        if mime.endswith("/webp"):
            return "webp"
        if "/" in mime:
            return mime.split("/", 1)[1]
    if fallback == "jpeg":
        return "jpg"
    return fallback


def _write_image(out_dir: Path, provider: str, index: int, image: ProviderImage, output_format: str, stamp: str) -> Path:
    ext = _extension_from_mime(image.mime_type, output_format)
    filename = f"{provider}-{stamp}-{index:02d}.{ext}"
    path = out_dir / filename
    path.write_bytes(image.image_bytes)
    return path


def _finalize_results(
    *,
    request: ImageRequest,
    resolved_provider: str,
    resolved,
    provider_response,
    out_dir: Path,
    stamp: str,
) -> List[ImageResult]:
    results: List[ImageResult] = []
    for idx, image in enumerate(provider_response.images):
        image_path = _write_image(out_dir, resolved_provider, idx, image, resolved.output_format, stamp)
        receipt_path = out_dir / f"receipt-{resolved_provider}-{stamp}-{idx:02d}.json"
        receipt_payload = build_receipt(
            request=request,
            resolved=resolved,
            provider_request=provider_response.raw_request,
            provider_response=provider_response.raw_response,
            warnings=list(resolved.warnings),
            image_path=image_path,
            receipt_path=receipt_path,
            result_metadata={
                "provider_metadata": image.metadata,
                "provider_request_id": image.provider_request_id,
            },
        )
        write_receipt(receipt_path, receipt_payload)
        results.append(
            ImageResult(
                image_path=image_path,
                receipt_path=receipt_path,
                provider=resolved_provider,
                model=provider_response.model,
                provider_request_id=image.provider_request_id,
                width=image.width,
                height=image.height,
                seed=image.seed,
                usage=provider_response.usage,
                metadata=image.metadata,
            )
        )
    return results


def generate(
    *,
    prompt: str,
    size: str = "1024x1024",
    n: int = 1,
    seed: Optional[int] = None,
    output_format: Optional[str] = None,
    background: Optional[str] = None,
    provider: Optional[str] = None,
    provider_options: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
    user: Optional[str] = None,
    model: Optional[str] = None,
) -> List[ImageResult]:
    request = ImageRequest(
        prompt=prompt,
        mode="generate",
        size=size,
        n=n,
        seed=seed,
        output_format=output_format,
        background=background,
        provider=provider,
        provider_options=provider_options or {},
        user=user,
        out_dir=Path(out_dir) if out_dir else None,
        model=model,
    )
    resolved_provider = resolve_provider(provider)
    out_path = ensure_out_dir(request.out_dir)
    request.out_dir = out_path
    resolved = resolve_request(request, resolved_provider)
    adapter = get_adapter(resolved_provider)
    response = adapter.generate(request, resolved)
    stamp = utc_timestamp()
    return _finalize_results(
        request=request,
        resolved_provider=resolved_provider,
        resolved=resolved,
        provider_response=response,
        out_dir=out_path,
        stamp=stamp,
    )


def edit(
    *,
    prompt: str,
    init_image,
    mask: Optional[object] = None,
    size: str = "1024x1024",
    n: int = 1,
    seed: Optional[int] = None,
    output_format: Optional[str] = None,
    background: Optional[str] = None,
    provider: Optional[str] = None,
    provider_options: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
    user: Optional[str] = None,
    model: Optional[str] = None,
) -> List[ImageResult]:
    inputs = ImageInputs(init_image=init_image, mask=mask)
    request = ImageRequest(
        prompt=prompt,
        mode="edit",
        size=size,
        n=n,
        seed=seed,
        output_format=output_format,
        background=background,
        inputs=inputs,
        provider=provider,
        provider_options=provider_options or {},
        user=user,
        out_dir=Path(out_dir) if out_dir else None,
        model=model,
    )
    resolved_provider = resolve_provider(provider)
    out_path = ensure_out_dir(request.out_dir)
    request.out_dir = out_path
    resolved = resolve_request(request, resolved_provider)
    adapter = get_adapter(resolved_provider)
    response = adapter.generate(request, resolved)
    stamp = utc_timestamp()
    return _finalize_results(
        request=request,
        resolved_provider=resolved_provider,
        resolved=resolved,
        provider_response=response,
        out_dir=out_path,
        stamp=stamp,
    )


def stream(
    *,
    prompt: str,
    size: str = "1024x1024",
    n: int = 1,
    seed: Optional[int] = None,
    output_format: Optional[str] = None,
    background: Optional[str] = None,
    provider: Optional[str] = None,
    provider_options: Optional[dict] = None,
    out_dir: Optional[str | Path] = None,
    user: Optional[str] = None,
    model: Optional[str] = None,
    partial_images: Optional[int] = None,
) -> Iterator[ImageEvent]:
    request = ImageRequest(
        prompt=prompt,
        mode="generate",
        size=size,
        n=n,
        seed=seed,
        output_format=output_format,
        background=background,
        provider=provider,
        provider_options=provider_options or {},
        user=user,
        out_dir=Path(out_dir) if out_dir else None,
        stream=True,
        partial_images=partial_images,
        model=model,
    )
    resolved_provider = resolve_provider(provider)
    out_path = ensure_out_dir(request.out_dir)
    request.out_dir = out_path
    resolved = resolve_request(request, resolved_provider)
    adapter = get_adapter(resolved_provider)
    stamp = utc_timestamp()

    if resolved.stream:
        for event in adapter.stream(request, resolved):
            if event.type == "partial" and event.image_bytes is not None:
                yield ImageEvent(type="partial", index=event.index, image_bytes=event.image_bytes)
            elif event.type == "log":
                yield ImageEvent(type="log", index=event.index, message=event.message, data=event.data)
            elif event.type == "error":
                yield ImageEvent(type="error", index=event.index, message=event.message)
                return
            elif event.type == "final" and event.image is not None:
                provider_response = ProviderResponse(
                    images=[event.image],
                    model=resolved.model,
                    usage=event.image.metadata.get("usage") if event.image.metadata else None,
                    raw_request={},
                    raw_response={},
                    warnings=resolved.warnings,
                )
                results = _finalize_results(
                    request=request,
                    resolved_provider=resolved_provider,
                    resolved=resolved,
                    provider_response=provider_response,
                    out_dir=out_path,
                    stamp=stamp,
                )
                if results:
                    yield ImageEvent(type="final", index=event.index, result=results[0])
        return

    results = generate(
        prompt=prompt,
        size=size,
        n=n,
        seed=seed,
        output_format=output_format,
        background=background,
        provider=provider,
        provider_options=provider_options,
        out_dir=out_dir,
        user=user,
        model=model,
    )
    yield from wrap_final_results(results)
