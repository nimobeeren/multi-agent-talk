import asyncio
import os

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_trace_processors,
)
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from .tracing import instrument_openai_agents

load_dotenv()

# Use Azure OpenAI as LLM provider
openai = AsyncAzureOpenAI(
    # Responses API needs base_url instead of azure_endpoint
    base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1"
)
set_default_openai_client(openai)
# Disable default OpenAI tracing
set_trace_processors([])
# Use OpenTelemetry for tracing
instrument_openai_agents()


@function_tool
def add_numbers(a: int, b: int) -> int:
    return a + b


async def main():
    agent = Agent(
        name="Single Agent",
        instructions="You are a helpful assistant",
        tools=[add_numbers],
        model="gpt-4.1-mini",
    )

    result = await Runner.run(agent, "What is 2+2?")

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
