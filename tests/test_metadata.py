"""Tests for Message.metadata: open-ended runtime payload, string-only dump."""

import json

from pydantic import BaseModel

from pydantic_mas._message import Message, MessageType


class _Schema(BaseModel):
    a: int
    b: str


class TestMetadataFieldExists:
    def test_defaults_to_empty_dict(self):
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        assert msg.metadata == {}

    def test_accepts_arbitrary_objects_at_runtime(self):
        live = _Schema(a=1, b="hello")
        msg = Message(
            sender="a",
            receiver="b",
            type=MessageType.REQUEST,
            content="hi",
            metadata={"output_schema": _Schema, "live": live, "n": 7},
        )
        # Identity preserved at runtime — no copy / no validation coercion.
        assert msg.metadata["output_schema"] is _Schema
        assert msg.metadata["live"] is live
        assert msg.metadata["n"] == 7


class TestMetadataSerialization:
    def test_dump_renders_metadata_as_string(self):
        live = _Schema(a=1, b="hello")
        msg = Message(
            sender="a",
            receiver="b",
            type=MessageType.REQUEST,
            content="hi",
            metadata={"live": live, "schema": _Schema},
        )
        dumped = msg.model_dump()
        assert isinstance(dumped["metadata"], str)
        # Must contain *some* representation of the values — we don't pin the
        # exact format, just that nothing crashes on un-JSON-able objects.
        assert "live" in dumped["metadata"] or "_Schema" in dumped["metadata"]

    def test_json_dump_does_not_raise_on_non_jsonable_values(self):
        """The whole point of string-dumping: classes / live objects are fine."""
        msg = Message(
            sender="a",
            receiver="b",
            type=MessageType.REQUEST,
            content="hi",
            metadata={"schema": _Schema, "obj": _Schema(a=1, b="x")},
        )
        # Should not raise.
        s = msg.model_dump_json()
        # The metadata key is present in the JSON output.
        parsed = json.loads(s)
        assert "metadata" in parsed
        assert isinstance(parsed["metadata"], str)

    def test_empty_metadata_dumps_cleanly(self):
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        dumped = msg.model_dump()
        # An empty dict rendered as a string is fine — assert it's a string.
        assert isinstance(dumped["metadata"], str)
