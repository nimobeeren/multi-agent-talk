import asyncio
import os

from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_trace_processors,
)
from dotenv import load_dotenv
from exa_py import Exa
from openai import AsyncAzureOpenAI

from .crm_agent import crm_agent
from .tracing import instrument_openai_agents

load_dotenv()

# Use Azure OpenAI as LLM provider
openai = AsyncAzureOpenAI(
    # Responses API needs base_url instead of azure_endpoint
    # base_url=f"{os.getenv('AZURE_OPENAI_ENDPOINT')}/openai/v1"
)
set_default_openai_client(openai)
# Use Chat Completions API instead of Responses API
set_default_openai_api("chat_completions")
# Disable default OpenAI tracing
set_trace_processors([])
# Use OpenTelemetry for tracing
instrument_openai_agents()

exa = Exa(api_key=os.getenv("EXA_API_KEY"))


@function_tool
def web_search(query: str) -> str:
    results = exa.search_and_contents(query=query, num_results=3)
    print("Sources:\n" + "\n".join([result.url for result in results.results]))
    return results.context  # type: ignore


async def main():
    agent = Agent(
        name="Single Agent",
        instructions="You are a thorough research assistant. Briefly answer the query.",
        tools=[web_search, crm_agent.as_tool(tool_name=None, tool_description=None)],
        model="gpt-5-mini",
    )

    result = await Runner.run(
        agent,
        "What are the orders for the customer 'Skyline Builders'?",
    )

    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
