"""Runtime mirror of the messages/service in `proto/rag.proto` (FtsSearch only, for now).

There is no `protoc` / `grpcio-tools` in this repo's dev or CI toolchain, so rather than commit
generated stubs nobody can regenerate, the descriptor is built here with `descriptor_pb2` and
turned into real protobuf message classes. Wire format is identical to what protoc would emit
for `proto/rag.proto`. `tests/test_search_grpc.py` parses the .proto file and asserts this
mirror matches it field-for-field, so the two cannot drift silently. When protoc lands, delete
this module and import the generated code instead; callers only use the names exported here.
"""

from __future__ import annotations

from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PACKAGE = "mytheus.intelligence.rag.v1"
SERVICE_NAME = f"{PACKAGE}.FtsService"
FTS_SEARCH_METHOD = f"/{SERVICE_NAME}/FtsSearch"

_F = descriptor_pb2.FieldDescriptorProto

# message -> [(name, number, type, repeated, type_name)]
_MESSAGES: dict[str, list[tuple[str, int, int, bool, str | None]]] = {
    "FtsSearchRequest": [
        ("query", 1, _F.TYPE_STRING, False, None),
        ("meeting_id", 2, _F.TYPE_STRING, False, None),
        ("limit", 3, _F.TYPE_INT32, False, None),
    ],
    "FtsHit": [
        ("passage", 1, _F.TYPE_STRING, False, None),
        ("rank", 2, _F.TYPE_DOUBLE, False, None),
        ("chunk_id", 3, _F.TYPE_STRING, False, None),
        ("meeting_id", 4, _F.TYPE_STRING, False, None),
        ("speaker_id", 5, _F.TYPE_STRING, False, None),
        ("start_ms", 6, _F.TYPE_INT64, False, None),
        ("end_ms", 7, _F.TYPE_INT64, False, None),
    ],
    "FtsSearchResponse": [
        ("query", 1, _F.TYPE_STRING, False, None),
        ("results", 2, _F.TYPE_MESSAGE, True, f".{PACKAGE}.FtsHit"),
    ],
}


def _build() -> dict[str, Any]:
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "mytheus_intelligence_rag_v1_fts.proto"
    fdp.package = PACKAGE
    fdp.syntax = "proto3"
    for msg_name, fields in _MESSAGES.items():
        m = fdp.message_type.add()
        m.name = msg_name
        for name, number, ftype, repeated, type_name in fields:
            f = m.field.add()
            f.name, f.number, f.type = name, number, ftype
            f.label = _F.LABEL_REPEATED if repeated else _F.LABEL_OPTIONAL
            if type_name:
                f.type_name = type_name
    svc = fdp.service.add()
    svc.name = "FtsService"
    method = svc.method.add()
    method.name = "FtsSearch"
    method.input_type = f".{PACKAGE}.FtsSearchRequest"
    method.output_type = f".{PACKAGE}.FtsSearchResponse"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{PACKAGE}.{name}"))
        for name in _MESSAGES
    }


_classes = _build()
FtsSearchRequest: Any = _classes["FtsSearchRequest"]
FtsHit: Any = _classes["FtsHit"]
FtsSearchResponse: Any = _classes["FtsSearchResponse"]
