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
from exa_py import Exa
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

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


@function_tool
def web_search(query: str) -> str:
    return exa.search_and_contents(query=query, num_results=3, context=True)  # type: ignore


async def main():
    agent = Agent(
        name="Single Agent",
        instructions="You are a thorough research assistant. Generate a one-page report based on the user's query.",
        tools=[web_search],
        model="gpt-5-mini",
    )

    result = await Runner.run(agent, "What are the differences between GPT-5 models available in ChatGPT and the API?")

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())

# NEXT UP
# create some "agents" which can answer questions using data from a CRM/PIM
# then try to answer questions which need at least 2 of these data sources
