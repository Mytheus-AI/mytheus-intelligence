"""serve_metrics binds a port once per process, however many consumer threads call it."""

import socket

from libs.shared import metrics


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_a_second_call_on_the_same_port_is_a_no_op() -> None:
    port = _free_port()
    metrics.serve_metrics(port)
    metrics.serve_metrics(port)  # used to raise OSError: Address already in use
    assert port in metrics._served_ports


def test_zero_disables() -> None:
    before = set(metrics._served_ports)
    metrics.serve_metrics(0)
    assert metrics._served_ports == before
