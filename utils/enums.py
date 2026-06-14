from enum import Enum

class AgentStatus(Enum):
    IDLE = 'idle'
    PROCESSING = 'processing'
    WAITING = 'waiting'
    ERROR = 'error'

class AgentCapabilities(Enum):
    KEYWORD_EXTRACTION = 'keyword extraction'
    RELEVANCE = 'relevance extraction'
    SHORT_TERM = 'short term memory'
    LONG_TERM = 'long term memory'
    ANCHOR = 'anchor'
    SUMMERIZER = 'summerizer'
    TEMPORAL = 'temporal'

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
