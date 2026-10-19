"""Exact, application-injected course enrichment contracts; no provider dependencies."""

from dataclasses import dataclass
import re
from typing import Callable


def validate_version(version):
    """Require a canonical positive ASCII integer with at most nine digits."""
    if not isinstance(version, str) or re.fullmatch(r"[1-9][0-9]{0,8}", version) is None:
        raise ValueError("Invalid enrichment contract version")
    return version


@dataclass(frozen=True)
class EnrichmentContract:
    prompt_version: str
    response_version: str
    build_prompt: Callable
    validate_result: Callable


class EnrichmentContractRegistry:
    def __init__(self):
        self._contracts = {}

    def register(self, prompt_version, response_version, build_prompt, validate_result):
        pair = (validate_version(prompt_version), validate_version(response_version))
        if not callable(build_prompt) or not callable(validate_result):
            raise TypeError("Enrichment contract handlers must be callable")
        if pair in self._contracts:
            raise ValueError("Enrichment contract already registered")
        self._contracts[pair] = EnrichmentContract(*pair, build_prompt, validate_result)

    def resolve(self, prompt_version, response_version):
        pair = (validate_version(prompt_version), validate_version(response_version))
        return self._contracts.get(pair)

    def capabilities(self):
        """Return detached metadata for registered tuples, in numeric version order."""
        return [
            {"promptVersion": prompt, "responseVersion": response}
            for prompt, response in sorted(self._contracts, key=lambda pair: tuple(map(int, pair)))
        ]
