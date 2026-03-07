"""Tests for the Deep Draft pattern implementation."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from make_llmstxt.deep_draft import (
    DeepDraftConfig,
    DraftState,
    SimpleDraftCritic,
    build_drafter_prompt,
)
from make_llmstxt.critic import CriticResult


class TestDeepDraftConfig:
    """Tests for DeepDraftConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DeepDraftConfig()
        assert config.max_rounds == 3
        assert config.pass_threshold == 0.7
        assert config.drafter_model == "claude-sonnet-4-6"
        assert config.critic_model == "claude-sonnet-4-6"
        assert config.temperature == 0.3

    def test_custom_config(self):
        """Test custom configuration values."""
        config = DeepDraftConfig(
            max_rounds=5,
            pass_threshold=0.8,
            drafter_model="gpt-4o",
            critic_model="gpt-4o-mini",
            temperature=0.5,
        )
        assert config.max_rounds == 5
        assert config.pass_threshold == 0.8
        assert config.drafter_model == "gpt-4o"
        assert config.critic_model == "gpt-4o-mini"
        assert config.temperature == 0.5

    def test_config_validation(self):
        """Test configuration validation."""
        # max_rounds bounds
        with pytest.raises(ValueError):
            DeepDraftConfig(max_rounds=0)
        with pytest.raises(ValueError):
            DeepDraftConfig(max_rounds=11)

        # pass_threshold bounds
        with pytest.raises(ValueError):
            DeepDraftConfig(pass_threshold=-0.1)
        with pytest.raises(ValueError):
            DeepDraftConfig(pass_threshold=1.1)


class TestDraftState:
    """Tests for DraftState."""

    def test_default_state(self):
        """Test default state values."""
        state = DraftState()
        assert state.draft == ""
        assert state.critique is None
        assert state.round == 0
        assert state.agreed is False
        assert state.history == []

    def test_state_with_values(self):
        """Test state with custom values."""
        critique = CriticResult(
            passed=True,
            score=0.85,
            issues=[],
            suggestions=[],
        )
        state = DraftState(
            draft="Test content",
            critique=critique,
            round=2,
            agreed=True,
            history=[{"round": 1, "score": 0.7}],
        )
        assert state.draft == "Test content"
        assert state.critique.passed is True
        assert state.round == 2
        assert state.agreed is True
        assert len(state.history) == 1


class TestBuildDrafterPrompt:
    """Tests for build_drafter_prompt."""

    def test_initial_prompt(self):
        """Test prompt without feedback."""
        prompt = build_drafter_prompt(
            url="https://example.com",
            content="Page 1: About\nPage 2: Docs",
        )
        assert "https://example.com" in prompt
        assert "Page 1: About" in prompt
        assert "PREVIOUS ATTEMPT FAILED" not in prompt

    def test_revision_prompt(self):
        """Test prompt with feedback."""
        prompt = build_drafter_prompt(
            url="https://example.com",
            content="Page 1: About\nPage 2: Docs",
            feedback=["Missing H1 header", "Generic description"],
        )
        assert "https://example.com" in prompt
        assert "PREVIOUS ATTEMPT FAILED" in prompt
        assert "Missing H1 header" in prompt
        assert "Generic description" in prompt


class TestSimpleDraftCritic:
    """Tests for SimpleDraftCritic."""

    @pytest.fixture
    def mock_config(self):
        """Create a test configuration."""
        return DeepDraftConfig(
            max_rounds=2,
            pass_threshold=0.7,
            drafter_model="gpt-4o-mini",
        )

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        return MagicMock(content="# Test Project\n> A test project for testing.\n\n## Core\n- [Home](https://example.com): Main page.")

    @pytest.fixture
    def mock_critic_response(self):
        """Create a mock critic result."""
        return CriticResult(
            passed=True,
            score=0.85,
            issues=[],
            suggestions=[],
        )

    @pytest.mark.asyncio
    async def test_draft_generation(self, mock_config, mock_llm_response):
        """Test draft generation."""
        with patch("make_llmstxt.deep_draft.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            mock_instance.ainvoke = AsyncMock(return_value=mock_llm_response)
            mock_chat.return_value = mock_instance

            critic = SimpleDraftCritic(config=mock_config)

            result = await critic.draft(
                url="https://example.com",
                content="Test content",
            )

            assert "# Test Project" in result or "test" in result.lower()

    @pytest.mark.asyncio
    async def test_critique_passes(self, mock_config):
        """Test critique when draft passes."""
        with patch("make_llmstxt.deep_draft.ChatOpenAI") as mock_chat:
            mock_instance = MagicMock()
            # First call = evaluation, Second call = extraction
            mock_instance.ainvoke = AsyncMock(
                side_effect=[
                    MagicMock(content="The draft looks good. It has proper structure."),
                    MagicMock(content='{"passed": true, "score": 0.85, "issues": [], "suggestions": []}'),
                ]
            )
            mock_chat.return_value = mock_instance

            critic = SimpleDraftCritic(config=mock_config)

            result = await critic.critique(
                llmstxt="# Test\n> A test.",
                url="https://example.com",
            )

            assert result.passed is True
            assert result.score >= 0.7
