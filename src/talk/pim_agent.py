from agents import Agent, ModelSettings, function_tool
from openai.types import Reasoning

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
    Get information from the Product Information Management system.
    Do not rely on your own knowledge, instead you may use tools.
    """,
    tools=[list_products],
    model="gpt-5-mini",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)
