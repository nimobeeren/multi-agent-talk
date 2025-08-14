from agents import Agent, function_tool

from .crm import db


@function_tool
def query_crm(query: str) -> str:
    """Query the CRM system for information. In this demo you'll always receive static order data regardless of the query."""
    return db


crm_agent = Agent(
    name="CRM Agent",
    instructions="You retrieve information from the CRM system.",
    tools=[query_crm],
    model="gpt-5-mini",
)
