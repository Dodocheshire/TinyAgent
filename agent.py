from llm import LLM
from trajectory import Trajectory

class TinyAgent:
    """极简、模块化智能体框架"""
    def __init__(self, llm: LLM):
        self.llm = llm
        self.memory = None
        self.tools = None
        self.planner = None
        self.trajectory = Trajectory()

    def run(self, task: str) -> str:
        """在给定任务上运行智能体"""
        self.trajectory.initialize(task)
        return self._step(task)

    def _step(self, task: str) -> str:
        """执行单个步骤"""
        messages = [{"role": "user", "content": task}]
        response = self.llm.generate(messages)
        self.trajectory.add(response)
        return response.content

    def _execute_action(self, action: str) -> str | None:
        """执行工具动作"""
        return f"Executed action: {action}"