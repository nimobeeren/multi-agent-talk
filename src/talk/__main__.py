import asyncio
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_trace_processors,
    # enable_verbose_stdout_logging,
)
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI

from .tracing import instrument_openai_agents

load_dotenv()
# enable_verbose_stdout_logging()

# Initialize the OpenAI client
openai = AsyncAzureOpenAI()
set_default_openai_client(openai)
# Use the chat completions API because Azure support for the Responses API is in preview
set_default_openai_api("chat_completions")
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
        model="gpt-4.1-mini"
    )

    result = await Runner.run(agent, "What is 2+2?")

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
