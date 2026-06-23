"""Return a configured Strands model client. Defaults to OpenAI for local POC;
swap to Bedrock by setting `MODEL_PROVIDER=bedrock` in the env."""
from __future__ import annotations

import os

from health_assistant.config import settings


def get_model():
    provider = os.environ.get("MODEL_PROVIDER", "openai").lower()

    if provider == "openai":
        from strands.models.openai import OpenAIModel

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set in environment")
        return OpenAIModel(
            client_args={"api_key": settings.openai_api_key},
            model_id=settings.openai_model,
        )

    if provider == "bedrock":
        from strands.models.bedrock import BedrockModel

        model_id = os.environ.get("BEDROCK_MODEL_ID")
        if not model_id:
            raise RuntimeError(
                "MODEL_PROVIDER=bedrock but BEDROCK_MODEL_ID is not set. Set it to a "
                "model/inference-profile you have access to, e.g. "
                "'us.anthropic.claude-3-5-sonnet-20241022-v2:0'. AWS credentials are "
                "read from the standard chain (env vars, SSO/temp creds, or a profile)."
            )
        return BedrockModel(
            model_id=model_id,
            region_name=os.environ.get("BEDROCK_REGION", "us-east-1"),
        )

    raise ValueError(
        f"Unknown MODEL_PROVIDER={provider!r}. Use 'openai' or 'bedrock'."
    )
