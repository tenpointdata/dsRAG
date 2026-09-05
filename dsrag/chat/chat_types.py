from typing import Optional
from typing_extensions import TypedDict
from pydantic import BaseModel

# One definition of a metadata filter, shared with the vector stores that
# have to execute it. Re-exported here so `chat` callers keep importing it
# from the module they already use.
from dsrag.database.vector.types import MetadataFilter

class ChatThreadParams(TypedDict):
    kb_ids: Optional[list[str]]
    model: Optional[str]
    temperature: Optional[float]
    system_message: Optional[str]
    auto_query_model: Optional[str]
    auto_query_guidance: Optional[str]
    rse_params: Optional[dict]
    target_output_length: Optional[str]
    max_chat_history_tokens: Optional[int]

class ChatResponseOutput(TypedDict):
    response: str
    metadata: dict

class ChatResponseInput(BaseModel):
    user_input: str
    chat_thread_params: Optional[ChatThreadParams] = None
    metadata_filter: Optional[MetadataFilter] = None