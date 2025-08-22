from agents import Agent, function_tool

from .crm import db


@function_tool
def list_orders(date: str | None, customer: str | None, product_sku: str | None) -> str:
    """List orders given a query. In this demo you'll always receive static order data regardless of the query."""
    return db


crm_agent = Agent(
    name="CRM Agent",
    instructions="""
    You retrieve information from the Customer Relationship Management system.
    Briefly answer the query using the tools provided.
    Do not rely on your own knowledge.
    If you are asked to do something you can't do with your tools, explain why.
    """,
    tools=[list_orders],
    model="gpt-5-mini",
)
