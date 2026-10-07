# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The served model an evaluation runs against: an OpenAI-compatible endpoint and its tokenizer."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class OpenAIEndpoint:
    """OpenAI-compatible HTTP endpoint for a served model."""

    base_url: str = field(repr=False)
    model: str
    api_key: str | None = field(default=None, repr=False)

    def url(self, path: str) -> str:
        """Return an endpoint URL under the API root."""
        return f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"


@dataclass(frozen=True)
class RunningModel:
    """A model that is already being served by a launcher-owned backend."""

    endpoint: OpenAIEndpoint
    tokenizer: str | None = None
