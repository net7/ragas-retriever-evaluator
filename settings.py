from pydantic import BaseModel, Field
from cat.mad_hatter.decorators import plugin
from enum import Enum

class Language(str, Enum):
    English = "English"
    French = "French"
    German = "German"
    Italian = "Italian"
    Spanish = "Spanish"
    Russian = "Russian"
    Chinese = "Chinese"
    Japanese = "Japanese"
    Korean = "Korean"

class RagasEvaluatorSettings(BaseModel):
    language: Language = Field(
        default=Language.English,
        title="Plugin Language",
        description="The language for the plugin's user-facing messages.",
    )
    openai_api_key: str = Field(
        default="",
        title="OpenAI API Key for Judge",
        description="API key for the OpenAI model used as a judge by Ragas. This is required for the evaluation.",
    )
    judge_model: str = Field(
        default="gpt-3.5-turbo",
        title="Judge LLM Model",
        description="The name of the language model used by Ragas to judge the answers (e.g., 'gpt-4-turbo', 'gpt-3.5-turbo')."
    )
    judge_temperature: float = Field(
        default=0.0,
        title="Judge Temperature",
        description="The temperature for the judge model (0.0 for deterministic results).",
        ge=0.0,
        le=1.0
    )
    retrieval_k: int = Field(
        default=5,
        title="Retrieval K",
        description="The number of documents (k) to retrieve from declarative memory for each question.",
        ge=1
    )
    generation_prompt_template: str = Field(
        default="Based *only* on the following context, answer the question. Do not use any other information.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        title="Generation Prompt Template",
        description="Template for generating answers during evaluation. Use {context} and {question} as placeholders. Keep the constraint for faithful evaluation."
    )

@plugin
def settings_model():
    return RagasEvaluatorSettings 