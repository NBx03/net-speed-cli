"""Тесты для speedcheck."""

from collections.abc import Iterator

import httpx
import pytest

from speedcheck import (
    BenchmarkSummary,
    RequestResult,
    calculate_summary,
    format_bytes,
    main,
    run_benchmark,
    validate_url,
)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://example.com", "http://example.com"),
        ("https://example.com", "https://example.com"),
        ("https://example.com/download/file.bin", "https://example.com/download/file.bin"),
        ("https://example.com:8080/path?arg=1", "https://example.com:8080/path?arg=1"),
        ("  https://example.com/spaces  ", "https://example.com/spaces"),
    ],
)
def test_validate_url_valid(url: str, expected: str) -> None:
    assert validate_url(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com",
        "example.com",
        "",
        "   ",
        "http://",
        "https://",
        "http:///path",
        "file:///path/to/file",
    ],
)
def test_validate_url_invalid(url: str) -> None:
    with pytest.raises(ValueError, match="Некорректный URL"):
        validate_url(url)


def test_calculate_summary_math() -> None:
    # r1: 1 MB за 0.1 с (скорость 10 MB/s)
    # r2: 1 MB за 0.9 с (скорость 1.11 MB/s)
    # Среднее арифметическое скоростей было бы (10 + 1.11) / 2 = 5.55 MB/s.
    # Но правильный расчёт: total_bytes / total_seconds = 2 MB / 1.0 с = 2.0 MB/s.
    results = [
        RequestResult(index=1, bytes_downloaded=1_000_000, elapsed_seconds=0.1, status_code=200),
        RequestResult(index=2, bytes_downloaded=1_000_000, elapsed_seconds=0.9, status_code=200),
    ]

    summary = calculate_summary(results)
    assert summary.requests_count == 2
    assert summary.total_bytes == 2_000_000
    assert summary.total_seconds == pytest.approx(1.0)
    assert summary.avg_seconds == pytest.approx(0.5)

    assert summary.speed_mb_per_s == pytest.approx(2.0)
    assert summary.speed_mbps == pytest.approx(16.0)
    assert summary.speed_mbps == pytest.approx(summary.speed_mb_per_s * 8)


def test_calculate_summary_zero_seconds() -> None:
    results = [
        RequestResult(index=1, bytes_downloaded=1000, elapsed_seconds=0.0, status_code=200),
    ]
    summary = calculate_summary(results)
    assert summary.total_seconds == 0.0
    assert summary.speed_mb_per_s == 0.0
    assert summary.speed_mbps == 0.0


def test_calculate_summary_empty() -> None:
    with pytest.raises(ValueError, match="пуст"):
        calculate_summary([])


@pytest.mark.parametrize(
    ("num_bytes", "expected"),
    [
        (0, "0 B"),
        (500, "500 B"),
        (1500, "1.50 KB"),
        (2_500_000, "2.50 MB"),
        (1_250_000_000, "1.25 GB"),
    ],
)
def test_format_bytes(num_bytes: int, expected: str) -> None:
    assert format_bytes(num_bytes) == expected


def test_run_benchmark_sequential_and_payload() -> None:
    payload = b"A" * 10_000
    call_count = 0
    in_flight = 0
    max_in_flight = 0
    captured_headers: list[httpx.Headers] = []

    class StreamingBody(httpx.SyncByteStream):
        def __iter__(self) -> Iterator[bytes]:
            try:
                yield payload[:5000]
                yield payload[5000:]
            finally:
                nonlocal in_flight
                in_flight -= 1

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count, in_flight, max_in_flight
        call_count += 1
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        captured_headers.append(request.headers)
        return httpx.Response(200, stream=StreamingBody())

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        results, summary = run_benchmark(
            url="http://example.com/test.bin",
            requests_count=10,
            client=client,
        )

    assert call_count == 10
    assert max_in_flight == 1
    assert in_flight == 0
    assert len(results) == 10
    assert summary.total_bytes == 10 * len(payload)

    for headers in captured_headers:
        assert headers.get("Accept-Encoding") == "identity"
        assert "no-cache" in headers.get("Cache-Control", "")


def test_run_benchmark_redirect() -> None:
    payload = b"redirected_content"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"Location": "http://example.com/target"})
        return httpx.Response(200, stream=httpx.ByteStream(payload))

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, follow_redirects=True) as client:
        results, summary = run_benchmark(
            url="http://example.com/start",
            requests_count=2,
            client=client,
        )

    assert len(results) == 2
    assert summary.total_bytes == 2 * len(payload)
    for res in results:
        assert res.status_code == 200


@pytest.mark.parametrize("status_code", [404, 500])
def test_run_benchmark_http_errors(status_code: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, stream=httpx.ByteStream(b"error"))

    transport = httpx.MockTransport(handler)
    with (
        httpx.Client(transport=transport, follow_redirects=True) as client,
        pytest.raises(httpx.HTTPStatusError) as exc_info,
    ):
        run_benchmark(
            url="http://example.com/fail",
            requests_count=1,
            client=client,
        )
    assert exc_info.value.response.status_code == status_code


def test_main_invalid_url(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["ftp://invalid-url.com"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Ошибка" in captured.err


def test_main_invalid_requests_count(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["http://example.com", "-n", "0"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Ошибка" in captured.err


def test_main_invalid_timeout(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["http://example.com", "-t", "-5"])
    captured = capsys.readouterr()
    assert exit_code == 2
    assert "Ошибка" in captured.err


def test_main_http_status_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(*args, **kwargs):
        req = httpx.Request("GET", "http://example.com/error")
        resp = httpx.Response(404, request=req)
        raise httpx.HTTPStatusError("Not Found", request=req, response=resp)

    monkeypatch.setattr("speedcheck.run_benchmark", fake_run)
    exit_code = main(["http://example.com/error"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Ошибка HTTP 404" in captured.err


def test_main_request_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(*args, **kwargs):
        req = httpx.Request("GET", "http://example.com/timeout")
        raise httpx.ConnectTimeout("Таймаут подключения", request=req)

    monkeypatch.setattr("speedcheck.run_benchmark", fake_run)
    exit_code = main(["http://example.com/timeout"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "Сетевая ошибка" in captured.err


def test_main_keyboard_interrupt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_run(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("speedcheck.run_benchmark", fake_run)
    exit_code = main(["http://example.com"])
    captured = capsys.readouterr()
    assert exit_code == 130
    assert "Прервано пользователем." in captured.err


def test_main_success_verbose_and_quiet(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_result = RequestResult(1, 1_000_000, 0.2, 200)
    sample_summary = BenchmarkSummary(1, 1_000_000, 0.2, 0.2, 40.0, 5.0)

    def fake_run(url, requests_count, timeout, on_progress=None):
        if on_progress:
            on_progress(sample_result, requests_count)
        return [sample_result], sample_summary

    monkeypatch.setattr("speedcheck.run_benchmark", fake_run)

    # Обычный режим с прогрессом
    exit_code = main(["http://example.com", "-n", "1"])
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "[1/1] 1.00 MB за 0.20 с" in out
    assert "Результаты замера (1 запрос):" in out

    # Режим --quiet
    exit_code_quiet = main(["http://example.com", "-n", "1", "-q"])
    out_quiet = capsys.readouterr().out
    assert exit_code_quiet == 0
    assert "[1/1]" not in out_quiet
    assert "Результаты замера (1 запрос):" in out_quiet


def test_main_zero_bytes_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sample_result = RequestResult(1, 0, 0.1, 200)
    sample_summary = BenchmarkSummary(1, 0, 0.1, 0.1, 0.0, 0.0)

    monkeypatch.setattr(
        "speedcheck.run_benchmark", lambda *a, **kw: ([sample_result], sample_summary)
    )
    exit_code = main(["http://example.com", "-q"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "пустое тело (0 байт)" in captured.err
