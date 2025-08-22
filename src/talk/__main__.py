import asyncio
import sys

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    Runner,
    run_demo_loop,
)
from dotenv import load_dotenv

from .guardrails import construction_guardrail
from .crm_agent import crm_agent
from .pim_agent import pim_agent
from .tracing import instrument_openai_agents

load_dotenv()

# Instrument the Agents SDK to get some logging
instrument_openai_agents()

agent = Agent(
    name="Customer Service Agent",
    instructions="""
    You are a helpful and kind customer service agent for a construction materials company.
    Do not rely on your own knowledge, instead you may use tools.
    Do not offer to do things you can't do with your tools.
    """,
    tools=[
        crm_agent.as_tool(
            tool_name=None, tool_description="Can list orders and nothing else"
        ),
        pim_agent.as_tool(
            tool_name=None, tool_description="Can list products and nothing else"
        ),
    ],
    input_guardrails=[construction_guardrail],
    model="gpt-5-mini",
)


async def main():
    if len(sys.argv) > 1:
        result = await Runner.run(agent, sys.argv[1])
        print(result.final_output)
    else:
        try:
            await run_demo_loop(agent, stream=False)
        except InputGuardrailTripwireTriggered as e:
            print(
                f"Sorry, I can't help with that. Reason: {e.guardrail_result.output.output_info.reason}"
            )


if __name__ == "__main__":
    asyncio.run(main())
