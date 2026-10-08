import json
import io
import pytest

from server.server import NexLMRequestHandler, NexLMServerState, create_demo_server


class MockSocket:
    def __init__(self, data: bytes = b""):
        self.rfile = io.BytesIO(data)
        self.wfile = io.BytesIO()

    def makefile(self, mode, *args, **kwargs):
        if "r" in mode:
            return self.rfile
        return self.wfile

    def sendall(self, data):
        self.wfile.write(data)


def test_server_initialization():
    server = create_demo_server(port=0)
    assert NexLMServerState.model is not None
    assert NexLMServerState.tokenizer is not None
    server.server_close()


def test_server_request_handler_mock():
    # Setup mock request
    mock_sock = MockSocket(b"GET /health HTTP/1.1\r\nHost: localhost\r\n\r\n")
    server = create_demo_server(port=0)

    try:
        handler = NexLMRequestHandler(mock_sock, ("127.0.0.1", 12345), server)
        response_bytes = mock_sock.wfile.getvalue()
        assert b"200 OK" in response_bytes
        assert b"healthy" in response_bytes
    finally:
        server.server_close()
