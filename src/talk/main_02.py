"""Use the demo loop, add tools and instrument the Agents SDK."""

import asyncio

from agents import Agent, function_tool, run_demo_loop
from dotenv import load_dotenv

from talk.crm import order_db
from talk.pim import product_db
from talk.logging import setup_logging

load_dotenv()
setup_logging()


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


agent = Agent(
    name="Customer Service Agent",
    instructions="""
    You are a helpful and kind customer service agent for a construction materials company.
    Do not rely on your own knowledge, instead rely on your tools.
    """,
    tools=[list_products, list_orders],
)


async def main():
    await run_demo_loop(agent)


if __name__ == "__main__":
    asyncio.run(main())
