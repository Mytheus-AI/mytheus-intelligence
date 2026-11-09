"""Runtime mirror of `proto/knowledge_graph.proto`, built the same way as `rag_proto.py`.

There is no `protoc` / `grpcio-tools` in this repo's toolchain, so the descriptor is built with
`descriptor_pb2`; the wire format is what protoc would emit. `tests/test_knowledge_graph_grpc.py`
parses the .proto file and asserts this mirror matches it field-for-field.
"""

from __future__ import annotations

from typing import Any

from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

PACKAGE = "mytheus.intelligence.knowledge_graph.v1"
SERVICE_NAME = f"{PACKAGE}.KnowledgeGraphService"
QUERY_TEMPLATE_METHOD = f"/{SERVICE_NAME}/QueryTemplate"

_F = descriptor_pb2.FieldDescriptorProto

_MESSAGES: dict[str, list[tuple[str, int, int, bool, str | None]]] = {
    "TemplateParam": [
        ("name", 1, _F.TYPE_STRING, False, None),
        ("value", 2, _F.TYPE_STRING, False, None),
    ],
    "QueryTemplateRequest": [
        ("template", 1, _F.TYPE_STRING, False, None),
        ("params", 2, _F.TYPE_MESSAGE, True, f".{PACKAGE}.TemplateParam"),
    ],
    "GraphNode": [
        ("label", 1, _F.TYPE_STRING, False, None),
        ("id", 2, _F.TYPE_STRING, False, None),
        ("role", 3, _F.TYPE_STRING, False, None),
        ("depth", 4, _F.TYPE_INT32, False, None),
        ("properties_json", 5, _F.TYPE_STRING, False, None),
    ],
    "QueryTemplateResponse": [
        ("template", 1, _F.TYPE_STRING, False, None),
        ("nodes", 2, _F.TYPE_MESSAGE, True, f".{PACKAGE}.GraphNode"),
    ],
}


def _build() -> dict[str, Any]:
    fdp = descriptor_pb2.FileDescriptorProto()
    fdp.name = "mytheus_intelligence_knowledge_graph_v1.proto"
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
    svc.name = "KnowledgeGraphService"
    method = svc.method.add()
    method.name = "QueryTemplate"
    method.input_type = f".{PACKAGE}.QueryTemplateRequest"
    method.output_type = f".{PACKAGE}.QueryTemplateResponse"

    pool = descriptor_pool.DescriptorPool()
    pool.Add(fdp)
    return {
        name: message_factory.GetMessageClass(pool.FindMessageTypeByName(f"{PACKAGE}.{name}"))
        for name in _MESSAGES
    }


_classes = _build()
TemplateParam: Any = _classes["TemplateParam"]
QueryTemplateRequest: Any = _classes["QueryTemplateRequest"]
GraphNode: Any = _classes["GraphNode"]
QueryTemplateResponse: Any = _classes["QueryTemplateResponse"]
