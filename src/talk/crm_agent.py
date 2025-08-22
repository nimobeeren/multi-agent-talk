from agents import Agent, ModelSettings, function_tool
from openai.types import Reasoning

from .crm import db


@function_tool
def list_orders(date: str | None, customer: str | None, product_sku: str | None) -> str:
    """List orders given a query. In this demo you'll always receive static order data regardless of the query."""
    return db


crm_agent = Agent(
    name="CRM Agent",
    instructions="""
    Get information from the Customer Relationship Management system.
    Do not rely on your own knowledge, instead you may use tools.
    """,
    tools=[list_orders],
    model="gpt-5-mini",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)
