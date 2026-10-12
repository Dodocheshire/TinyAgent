from dataclasses import dataclass
from llm import Response

@dataclass
class Step:
    """agent 轨迹中的单个步骤"""
    thought: str ="" # 思考过程：reasoning
    action: dict | None =None # 描述一次工具调用
    observation: str | None =None # 工具使用的输出
    answer: str | None =None # 模型给出的最终答案，代表智能体已经走到了本轮的尽头
    metadata: dict | None =None

class Trajectory:
    """将智能体的执行过程记录为一系列运行"""
    # 注意这里是一系列运行，而不是一次。一次运行对应一次查询，多次可能的action和observation，直到模型给出最终answer
    def __init__(self) -> None:
        self.runs: list[dict] = []

    def initialize(self, query: str) -> None:
        """用给定的查询注册一次新的运行"""
        self.runs.append({"query": query, "steps": []})

    def add(self, response: Response, observation: str | None =None) -> None:
        """根据LLM的一次Response记录一个步骤，可附带观测结果"""
        # 添加思考
        step = Step(response.reasoning or "", metadata=response.metadata)
        # 添加行动/观测，或者回答
        if observation is not None:
            step.action = response.tool_call
            step.observation = observation
        else:
            step.answer = response.content
        self.runs[-1]["steps"].append(step)
