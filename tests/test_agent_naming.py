"""Tests for agent naming in MAS traces."""

from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel

from pydantic_mas import MAS, AgentConfig, Budget


class TestAgentNaming:
    """MAS should set agent names from agent_id for tracing."""

    async def test_sets_name_when_none(self):
        """Agent with no name gets agent_id as name after MAS.run()."""
        agent = Agent(TestModel(custom_output_text="done"))
        assert agent.name is None

        mas = MAS(
            agents={"coordinator": AgentConfig(agent=agent)},
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        await mas.run("coordinator", "hello")

        assert agent.name == "coordinator"

    async def test_preserves_existing_name(self):
        """Agent with user-set name is not overwritten."""
        agent = Agent(TestModel(custom_output_text="done"), name="my-agent")
        assert agent.name == "my-agent"

        mas = MAS(
            agents={"coordinator": AgentConfig(agent=agent)},
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        await mas.run("coordinator", "hello")

        assert agent.name == "my-agent"

    async def test_multiple_agents_named(self):
        """All agents without names get their agent_id."""
        a = Agent(TestModel(custom_output_text="done"))
        b = Agent(TestModel(custom_output_text="done"))
        c = Agent(TestModel(custom_output_text="done"), name="custom")

        mas = MAS(
            agents={
                "alpha": AgentConfig(agent=a),
                "beta": AgentConfig(agent=b),
                "gamma": AgentConfig(agent=c),
            },
            budget=Budget(max_total_messages=10, timeout_seconds=5),
        )
        await mas.run("alpha", "hello")

        assert a.name == "alpha"
        assert b.name == "beta"
        assert c.name == "custom"  # preserved
