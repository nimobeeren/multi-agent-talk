from agents import Agent, function_tool

from .pim import db


@function_tool
def list_products(
    id: str | None,
    sku: str | None,
    name: str | None,
    price: float | None,
    stock: int | None,
) -> str:
    """List products given a query. In this demo you'll always receive static product data regardless of the query."""
    return db


pim_agent = Agent(
    name="PIM Agent",
    instructions="""
    You retrieve information from the Product Information Management system.
    Briefly answer the query using the tools provided.
    Do not rely on your own knowledge.
    If you are asked to do something you can't do with your tools, explain why.
    """,
    tools=[list_products],
    model="gpt-5-mini",
)
