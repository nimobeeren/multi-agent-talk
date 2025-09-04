"""Add a guardrail."""

import asyncio

from agents import (
    Agent,
    InputGuardrailTripwireTriggered,
    ModelSettings,
    function_tool,
    run_demo_loop,
    input_guardrail,
    RunContextWrapper,
    TResponseInputItem,
    GuardrailFunctionOutput,
    Runner,
)
from pydantic import BaseModel
from dotenv import load_dotenv
from openai.types import Reasoning

from talk.crm import order_db
from talk.pim import product_db
from talk.tracing import instrument_openai_agents

load_dotenv()
instrument_openai_agents()


class GuardrailOuput(BaseModel):
    is_triggered: bool
    reason: str


guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="Trigger if the input is not relevant to a customer service agent for a construction material company",
    output_type=GuardrailOuput,
)


@input_guardrail
async def guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_triggered,
    )


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
    Do not offer to do things you can't do with your tools.
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
    Do not offer to do things you can't do with your tools.
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
    Do not offer to do things you can't do with your tools.
    """,
    tools=[
        pim_agent.as_tool(
            tool_name=None, tool_description="Can list orders and nothing else"
        ),
        crm_agent.as_tool(
            tool_name=None, tool_description="Can list products and nothing else"
        ),
    ],
    input_guardrails=[guardrail],
    model="gpt-5-nano",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)


async def main():
    try:
        await run_demo_loop(agent)
    except InputGuardrailTripwireTriggered as e:
        print(
            f"Sorry, I can't help with that. {e.guardrail_result.output.output_info.reason}"
        )


if __name__ == "__main__":
    asyncio.run(main())
