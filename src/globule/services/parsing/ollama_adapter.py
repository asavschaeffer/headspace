from globule.core.interfaces import IParserProvider
from globule.core.errors import ParserError
from .ollama_parser import OllamaParser


class OllamaParsingAdapter(IParserProvider):
    def __init__(self, provider: OllamaParser):
        self._provider = provider
    
    async def parse(self, text: str) -> dict:
        try:
            result = await self._provider.parse(text)
            return result
        except (JSONDecodeError, ValidationError, PydanticError, ClientError, RuntimeError) as e:
            raise ParserError(f"Ollama parsing provider failed: {e}") from e