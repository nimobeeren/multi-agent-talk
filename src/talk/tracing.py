"""
Based on: https://langfuse.com/docs/integrations/openaiagentssdk/openai-agents
"""

import base64
import os

import logfire


def configure_otel():
    """Configure OpenTelemetry to send traces to Langfuse."""
    print("Configuring OpenTelemetry")
    langfuse_host = os.getenv("LANGFUSE_HOST")
    langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")

    if not langfuse_host or not langfuse_public_key or not langfuse_secret_key:
        print("Langfuse environment variables not set, skipping instrumentation")
        return

    # Build Basic Auth header
    langfuse_auth = base64.b64encode(
        f"{langfuse_public_key}:{langfuse_secret_key}".encode()
    ).decode()

    # Configure OpenTelemetry endpoint & headers
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = langfuse_host + "/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"


def instrument_openai_agents():
    """Instrument the OpenAI Agents SDK to use OpenTelemetry."""
    configure_otel()

    # Configure Logfire instrumentation
    print("Instrumenting OpenAI Agents")
    logfire.configure(
        service_name="multi_agent_talk",
        send_to_logfire=False,
        # By default, Logfire scrubs potentially sensitive data from traces. However,
        # that includes the session ID which is used by Langfuse to group traces.
        scrubbing=False,
    )
    # This method automatically patches the OpenAI Agents SDK to send logs via OTLP
    logfire.instrument_openai_agents()
