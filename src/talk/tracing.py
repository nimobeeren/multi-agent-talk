import logfire


def instrument_openai_agents():
    """Instrument the OpenAI Agents SDK to use OpenTelemetry."""
    print("Instrumenting OpenAI Agents")
    logfire.configure(service_name="multi_agent_talk", send_to_logfire=False)
    logfire.instrument_openai_agents()
