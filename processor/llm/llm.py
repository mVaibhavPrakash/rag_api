from typing import TypeDict, Literal

VariantType= Literal['Default', 'low', 'medium', 'high', 'xhigh', 'max']
CurrencyType= Literal['Dollar', 'Rupees']

class CostInfo:
    cost: int
    currency: str

class ModelInfo(TypeDict):
    model: str
    variant: list[VariantType]
    context_size: str
    input_cost: CostInfo


class LLMProvider(TypeDict):
    provider: str
    model: list[ModelInfo]
    host: str

def get_llm_list() -> ModelInfo:
    llm: list[LLMProvider] = [{
        "provider": 'openAI',

    }]
    return llm