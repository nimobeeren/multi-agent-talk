"""Basic agent."""

import asyncio

from agents import Agent, Runner, ModelSettings
from openai.types import Reasoning
from dotenv import load_dotenv

load_dotenv()

agent = Agent(
    name="Customer Service Agent",
    instructions="""
    You are a helpful and kind customer service agent for a construction materials company.
    """,
    model="gpt-5-nano",
    model_settings=ModelSettings(
        verbosity="low", reasoning=Reasoning(effort="minimal")
    ),
)


async def main():
    inp = input("Enter a message: ")
    result = await Runner.run(agent, inp)
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
