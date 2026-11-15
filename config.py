"""Validated JSON5 settings for the depth estimation node."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal, Self

import json5
import zenoh
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

NonEmptyStr = Annotated[StrictStr, Field(pattern=r"\S")]
PositiveInt = Annotated[int, Field(strict=True, ge=1)]


def _validate_concrete_key(key: str, field_name: str) -> str:
    try:
        zenoh.KeyExpr(key)
    except zenoh.ZError as error:
        raise ValueError(f"{field_name} must be a valid Zenoh key: {error}") from error
    if not key.strip() or "*" in key:
        raise ValueError(f"{field_name} must be a non-empty key without wildcards")
    return key


StreamName = Annotated[
    StrictStr,
    Field(pattern=r"^[^/]+$"),
    AfterValidator(partial(_validate_concrete_key, field_name="stream name")),
]
SourceKey = Annotated[
    StrictStr,
    AfterValidator(partial(_validate_concrete_key, field_name="source key")),
]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BaseEstimatorConfig(ConfigModel):
    model_id: NonEmptyStr
    revision: NonEmptyStr = "main"
    device: NonEmptyStr


class TransformersEstimatorConfig(BaseEstimatorConfig):
    backend: Literal["transformers"]
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"


class DepthAnything3EstimatorConfig(BaseEstimatorConfig):
    backend: Literal["depth_anything_3"]
    process_res: PositiveInt = 504


EstimatorConfig = Annotated[
    TransformersEstimatorConfig | DepthAnything3EstimatorConfig,
    Field(discriminator="backend"),
]


class PublisherConfig(ConfigModel):
    congestion_control: Literal["drop", "block", "block_first"]
    reliability: Literal["best_effort", "reliable"]


class NodeConfig(ConfigModel):
    zenoh_key_prefix: StrictStr
    estimator: EstimatorConfig
    streams: Annotated[
        Mapping[StreamName, SourceKey],
        Field(min_length=1),
        AfterValidator(MappingProxyType),
    ]
    publisher: PublisherConfig

    @field_validator("zenoh_key_prefix")
    @classmethod
    def validate_zenoh_key_prefix(cls, prefix: str) -> str:
        return _validate_concrete_key(prefix, "zenoh_key_prefix")

    @model_validator(mode="after")
    def validate_stream_keys(self) -> Self:
        source_keys = list(self.streams.values())
        if len(set(source_keys)) != len(source_keys):
            raise ValueError(f"stream source keys must be unique: {source_keys}")
        for name in self.streams:
            depth_key = f"{self.zenoh_key_prefix}/{name}"
            if depth_key in source_keys:
                raise ValueError(
                    f"depth key '{depth_key}' of stream '{name}' is also a source key"
                )
        return self


def load_node_config(path: Path) -> NodeConfig:
    try:
        document = json5.loads(
            path.read_text(encoding="utf-8"), allow_duplicate_keys=False
        )
    except ValueError as error:
        raise ValueError(f"{path}: invalid JSON5: {error}") from error
    try:
        return NodeConfig.model_validate(document)
    except ValidationError as error:
        raise ValueError(f"{path}: invalid node configuration:\n{error}") from error
