"""Tackle capability negotiation issue."""

import asyncio

from agents import Agent, function_tool, run_demo_loop
from dotenv import load_dotenv

from talk.crm import order_db
from talk.pim import product_db
from talk.tracing import instrument_openai_agents

load_dotenv()
instrument_openai_agents()


@function_tool
def list_products(
    sku: str | None,
    name: str | None,
    price: float | None,
    stock: int | None,
) -> str:
    return product_db


@function_tool
def list_orders(
    date: str | None,
    product_sku: str | None,
    quantity: int | None,
) -> str:
    return order_db


pim_agent = Agent(
    name="PIM Agent",
    instructions="""
    You retrieve information from the Product Information Management system.
    Do not rely on your own knowledge, instead rely on your tools.
    Do not offer to do things you can't do with your tools.
    """,
    tools=[list_products],
)

crm_agent = Agent(
    name="CRM Agent",
    instructions="""
    You retrieve information from the Customer Relationship Management system.
    Do not rely on your own knowledge, instead rely on your tools.
    Do not offer to do things you can't do with your tools.
    """,
    tools=[list_orders],
)

agent = Agent(
    name="Customer Service Agent",
    instructions="""
    You are a helpful and kind customer service agent for a construction materials company.
    Do not rely on your own knowledge, instead rely on your tools.
    Do not offer to do things you can't do with your tools.
    """,
    tools=[
        pim_agent.as_tool(
            tool_name=None, tool_description="Can list products and nothing else"
        ),
        crm_agent.as_tool(
            tool_name=None, tool_description="Can list orders and nothing else"
        ),
    ],
)


async def main():
    await run_demo_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
