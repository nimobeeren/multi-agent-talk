from agents import Agent, function_tool

from .pim import db


@function_tool
def query_pim(query: str) -> str:
    """Query the PIM system for information. In this demo you'll always receive static product data regardless of the query."""
    return db


pim_agent = Agent(
    name="PIM Agent",
    instructions="You retrieve information from the Product Information Management system.",
    tools=[query_pim],
    model="gpt-5-mini",
)
