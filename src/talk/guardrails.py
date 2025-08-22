from agents import (
    Agent,
    GuardrailFunctionOutput,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)
from pydantic import BaseModel


class ConstructionGuardrailOuput(BaseModel):
    is_triggered: bool
    reason: str


guardrail_agent = Agent(
    name="Construction Guardrail Agent",
    instructions="Trigger if the input is not relevant to a customer service agent for a construction material company",
    output_type=ConstructionGuardrailOuput,
)


@input_guardrail
async def construction_guardrail(
    ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)

    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_triggered,
    )
