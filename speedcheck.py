"""Утилита для замера скорости скачивания по URL (последовательные HTTP GET-запросы)."""

import argparse
import sys
import time
import urllib.parse
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import httpx

DEFAULT_REQUESTS = 10
DEFAULT_TIMEOUT = 30.0
CHUNK_SIZE = 65_536  # 64 KiB

# Accept-Encoding: identity отключает сжатие (gzip/br), чтобы замерять реальный размер данных.
# Cache-Control и Pragma запрещают промежуточным прокси и серверам отдавать кэш.
DEFAULT_HEADERS: dict[str, str] = {
    "Accept-Encoding": "identity",
    "Cache-Control": "no-cache, no-store",
    "Pragma": "no-cache",
}


@dataclass(frozen=True, slots=True)
class RequestResult:
    index: int
    bytes_downloaded: int
    elapsed_seconds: float
    status_code: int


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    requests_count: int
    total_bytes: int
    total_seconds: float
    avg_seconds: float
    speed_mbps: float
    speed_mb_per_s: float


def validate_url(url: str) -> str:
    cleaned = url.strip()
    parsed = urllib.parse.urlsplit(cleaned)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError(
            f"Некорректный URL '{url}'. Ожидается абсолютный адрес с протоколом http:// или https://."
        )
    return cleaned


def calculate_summary(results: Sequence[RequestResult]) -> BenchmarkSummary:
    if not results:
        raise ValueError("Список результатов измерений пуст.")

    total_bytes = sum(r.bytes_downloaded for r in results)
    total_seconds = sum(r.elapsed_seconds for r in results)
    avg_seconds = total_seconds / len(results)

    speed_mb_per_s = (total_bytes / 1_000_000) / total_seconds if total_seconds > 0 else 0.0
    speed_mbps = (total_bytes * 8 / 1_000_000) / total_seconds if total_seconds > 0 else 0.0

    return BenchmarkSummary(
        requests_count=len(results),
        total_bytes=total_bytes,
        total_seconds=total_seconds,
        avg_seconds=avg_seconds,
        speed_mbps=speed_mbps,
        speed_mb_per_s=speed_mb_per_s,
    )


def format_bytes(num_bytes: int) -> str:
    if num_bytes < 1000:
        return f"{num_bytes} B"
    val = float(num_bytes)
    for unit in ("KB", "MB", "GB"):
        val /= 1000.0
        if val < 1000.0 or unit == "GB":
            return f"{val:.2f} {unit}"
    return f"{val:.2f} GB"


def _format_requests_count(count: int) -> str:
    if 11 <= count % 100 <= 14:
        return f"{count} запросов"
    rem = count % 10
    if rem == 1:
        return f"{count} запрос"
    if 2 <= rem <= 4:
        return f"{count} запроса"
    return f"{count} запросов"


def measure_single_request(client: httpx.Client, url: str, index: int) -> RequestResult:
    start = time.perf_counter()
    with client.stream("GET", url, headers=DEFAULT_HEADERS) as response:
        response.raise_for_status()
        bytes_downloaded = 0
        # iter_raw вычитывает сырые байты из сокета без декодирования и без расхода памяти O(N).
        for chunk in response.iter_raw(chunk_size=CHUNK_SIZE):
            bytes_downloaded += len(chunk)
    elapsed_seconds = time.perf_counter() - start

    return RequestResult(
        index=index,
        bytes_downloaded=bytes_downloaded,
        elapsed_seconds=elapsed_seconds,
        status_code=response.status_code,
    )


def run_benchmark(
    url: str,
    requests_count: int = DEFAULT_REQUESTS,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.Client | None = None,
    on_progress: Callable[[RequestResult, int], None] | None = None,
) -> tuple[list[RequestResult], BenchmarkSummary]:
    if requests_count < 1:
        raise ValueError("Количество запросов должно быть не менее 1.")
    if timeout <= 0:
        raise ValueError("Таймаут должен быть больше 0.")

    def _execute(active_client: httpx.Client) -> list[RequestResult]:
        items: list[RequestResult] = []
        for index in range(1, requests_count + 1):
            result = measure_single_request(active_client, url, index)
            items.append(result)
            if on_progress is not None:
                on_progress(result, requests_count)
        return items

    if client is not None:
        results = _execute(client)
    else:
        timeout_config = httpx.Timeout(timeout, connect=min(10.0, timeout))
        with httpx.Client(
            follow_redirects=True,
            timeout=timeout_config,
        ) as new_client:
            results = _execute(new_client)

    return results, calculate_summary(results)


def print_summary(summary: BenchmarkSummary) -> None:
    req_label = _format_requests_count(summary.requests_count)
    bytes_str = f"{summary.total_bytes:_}".replace("_", " ")
    print(f"Результаты замера ({req_label}):")
    print(f"  Всего скачано:         {format_bytes(summary.total_bytes)} ({bytes_str} байт)")
    print(f"  Среднее время запроса: {summary.avg_seconds:.3f} с")
    print(
        f"  Скорость скачивания:   {summary.speed_mbps:.2f} Mbps "
        f"({summary.speed_mb_per_s:.2f} MB/s)"
    )
    if summary.total_bytes == 0:
        print("Внимание: сервер вернул пустое тело (0 байт).", file=sys.stderr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speedcheck",
        description="Замер скорости скачивания по URL (10 последовательных GET-запросов).",
    )
    parser.add_argument("url", help="URL ресурса для замера скорости")
    parser.add_argument(
        "-n",
        "--requests",
        type=int,
        default=DEFAULT_REQUESTS,
        help=f"Число запросов ({DEFAULT_REQUESTS})",
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"Таймаут в сек ({DEFAULT_TIMEOUT})",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Отключить вывод прогресса")

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
        url = validate_url(args.url)

        def on_progress(res: RequestResult, total: int) -> None:
            pad = len(str(total))
            size = format_bytes(res.bytes_downloaded)
            print(f"[{res.index:>{pad}}/{total}] {size} за {res.elapsed_seconds:.2f} с", flush=True)

        _, summary = run_benchmark(
            url=url,
            requests_count=args.requests,
            timeout=args.timeout,
            on_progress=None if args.quiet else on_progress,
        )

        if not args.quiet:
            print()
        print_summary(summary)
        return 0
    except ValueError as err:
        print(f"Ошибка: {err}", file=sys.stderr)
        return 2
    except httpx.HTTPStatusError as err:
        print(
            f"Ошибка HTTP {err.response.status_code} при обращении к {err.request.url}",
            file=sys.stderr,
        )
        return 1
    except httpx.RequestError as err:
        print(f"Сетевая ошибка при обращении к {err.request.url}: {err}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
