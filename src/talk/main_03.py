"""Make it multi-agent."""

import asyncio

from agents import Agent, ModelSettings, function_tool, run_demo_loop
from dotenv import load_dotenv
from openai.types import Reasoning

from talk.crm import order_db
from talk.pim import product_db
from talk.tracing import instrument_openai_agents

load_dotenv()
instrument_openai_agents()


@function_tool
def list_products(
    id: str | None,
    sku: str | None,
    name: str | None,
    price: float | None,
    stock: int | None,
) -> str:
    """List products given a query."""
    return product_db


@function_tool
def list_orders(
    date: str | None,
    customer: str | None,
    product_sku: str | None,
) -> str:
    """List orders given a query."""
    return order_db


pim_agent = Agent(
    name="PIM Agent",
    instructions="""
    You retrieve information from the Product Information Management system.
    Do not rely on your own knowledge, instead rely on your tools.
    """,
    tools=[list_products],
    model="gpt-5-nano",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)

crm_agent = Agent(
    name="CRM Agent",
    instructions="""
    You retrieve information from the Customer Relationship Management system.
    Do not rely on your own knowledge, instead rely on your tools.
    """,
    tools=[list_orders],
    model="gpt-5-nano",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)

agent = Agent(
    name="Customer Service Agent",
    instructions="""
    You are a helpful and kind customer service agent for a construction materials company.
    Do not rely on your own knowledge, instead rely on your tools.
    """,
    tools=[
        pim_agent.as_tool(tool_name=None, tool_description=None),
        crm_agent.as_tool(tool_name=None, tool_description=None),
    ],
    model="gpt-5-nano",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)


async def main():
    await run_demo_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
