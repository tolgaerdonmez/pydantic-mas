from pydantic_mas._message import Message, MessageType


def default_message_formatter(message: Message) -> str:
    """Default formatter: presents a message as a labeled block for the LLM."""
    if message.sender == "system":
        return f"[System | New task]\n{message.content}"

    if message.type == MessageType.REPLY:
        return (
            f"[Reply from '{message.sender}' | in reply to: {message.in_reply_to} "
            f"| id: {message.id}]\n{message.content}"
        )

    return (
        f"[Message from '{message.sender}' | type: {message.type.value} "
        f"| id: {message.id}]\n{message.content}"
    )
