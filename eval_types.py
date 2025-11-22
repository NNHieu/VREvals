from typing import Any, List, Dict, Optional
from pydantic import BaseModel, Field

Message = Dict[str, Any]  # keys role, content
MessageList = List[Message]

class ResponseChoice(BaseModel):
    response_text: str

class SamplerResponse(BaseModel):
    """
    Response from a sampler.
    """
    status: str
    actual_queried_message_list: MessageList | None
    response_metadata: Dict[str, Any] | None
    choices: List[ResponseChoice]

class PerGenerationEvalResult(BaseModel):
    generation: str
    pred_answer: Optional[str]
    metric: Optional[Dict[str, float]]

class SamplerBase:
    """
    Base class for defining a sampling model, which can be evaluated,
    or used as part of the grading process.
    """

    def __call__(
        self, 
        message_list: MessageList,
    ) -> SamplerResponse:
        raise NotImplementedError

class EvalResult(BaseModel):
    """
    Result of running an evaluation (usually consisting of many samples)
    """
    score: Optional[float]  # top-line metric
    metrics: Optional[Dict[str, float]]  # other metrics
    htmls: List[str]  # strings of valid HTML
    convos: List[MessageList]  # sampled conversations
    metadata: Optional[Dict[str, Any]]  # Extra data such as rubric scores or sollen

class SingleEvalResult(BaseModel):
    """
    Result of evaluating a single sample
    """
    score: Optional[float]
    metrics: Dict[str, float] = Field(default_factory=dict)
    html: Optional[str] = None
    convo: Optional[MessageList] = None  # sampled conversation
    example_level_metadata: Optional[Dict[str, Any]] = None  # Extra data such as rubric scores or sollen

class Eval:
    """
    Base class for defining an evaluation.
    """

    def __call__(self, sampler: SamplerBase) -> EvalResult:
        raise NotImplementedError
