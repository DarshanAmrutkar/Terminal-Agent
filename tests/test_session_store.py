"""Tests for session persistence and SessionStore."""

import tempfile

import pytest

from terminal_agent.core.session import Session
from terminal_agent.core.session_store import SessionStore
from terminal_agent.llm.message import Message, ToolCall, ToolResultContent


def test_session_to_dict_and_from_dict_roundtrip():
    """Verify full fidelity serialization and deserialization of a Session."""
    session = Session(working_directory="/test/project", session_id="test_session_123")
    session.set_system_message("You are an AI assistant.")
    session.add_user_message("Write a hello world program")
    
    tc = ToolCall(id="call_99", name="write_file", arguments={"path": "hello.py", "content": "print('hello')"})
    session.add_assistant_message(Message.assistant(content="Creating file", tool_calls=[tc]))
    
    tr = ToolResultContent(tool_call_id="call_99", output="Wrote 14 bytes to hello.py", is_error=False)
    session.add_tool_result_message(Message.tool_result([tr]))
    
    session.track_file("hello.py")
    session.update_token_usage(input_tokens=150, output_tokens=45)

    data = session.to_dict()
    assert data["session_id"] == "test_session_123"
    assert data["working_directory"] == "/test/project"
    assert data["total_input_tokens"] == 150
    assert data["total_output_tokens"] == 45
    assert "hello.py" in data["active_files"]
    assert len(data["messages"]) == 4

    reconstituted = Session.from_dict(data)
    assert reconstituted.session_id == session.session_id
    assert reconstituted.working_directory == session.working_directory
    assert reconstituted.total_input_tokens == 150
    assert reconstituted.total_output_tokens == 45
    assert reconstituted.active_files == session.active_files
    assert len(reconstituted.messages) == 4
    
    # Verify tool calls and results are intact
    assert reconstituted.messages[2].tool_calls[0].name == "write_file"
    assert reconstituted.messages[3].tool_results[0].output == "Wrote 14 bytes to hello.py"


def test_session_store_save_load_and_latest():
    """Verify saving, loading, prefix lookup, and getting latest session."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(storage_dir=temp_dir)
        
        # Save session 1
        s1 = Session(working_directory="/test/dir1", session_id="20260923_100000")
        s1.add_user_message("Initial request 1")
        store.save_session(s1)

        # Save session 2
        s2 = Session(working_directory="/test/dir2", session_id="20260923_110000")
        s2.add_user_message("Initial request 2")
        store.save_session(s2)

        # Load exact
        loaded1 = store.load_session("20260923_100000")
        assert loaded1 is not None
        assert loaded1.session_id == "20260923_100000"
        assert loaded1.messages[0].content == "Initial request 1"

        # Load by prefix
        loaded2 = store.load_session("20260923_11")
        assert loaded2 is not None
        assert loaded2.session_id == "20260923_110000"

        # Non-existent session
        assert store.load_session("non_existent_id") is None

        # Latest session (s2 was saved last)
        latest = store.get_latest_session()
        assert latest is not None
        assert latest.session_id == "20260923_110000"


def test_session_store_list_and_delete():
    """Verify session metadata listing and deletion."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(storage_dir=temp_dir)
        
        s = Session(working_directory="/test/work", session_id="sess_abc")
        s.add_user_message("How do I write unit tests in pytest?")
        s.update_token_usage(input_tokens=200, output_tokens=80)
        store.save_session(s)

        sessions_list = store.list_sessions(limit=5)
        assert len(sessions_list) == 1
        entry = sessions_list[0]
        assert entry["session_id"] == "sess_abc"
        assert "How do I write unit tests" in entry["preview"]
        assert entry["total_tokens"] == 280

        # Delete session
        deleted = store.delete_session("sess_abc")
        assert deleted is True
        assert store.load_session("sess_abc") is None
        assert len(store.list_sessions()) == 0


@pytest.mark.asyncio
async def test_agent_session_resumption():
    """Verify Agent can resume a session and preserve conversation history."""
    with tempfile.TemporaryDirectory() as temp_dir:
        store = SessionStore(storage_dir=temp_dir)
        
        prior_session = Session(working_directory=temp_dir, session_id="resumed_123")
        prior_session.set_system_message("System prompt")
        prior_session.add_user_message("Prior user message")
        prior_session.add_assistant_message(Message.assistant(content="Prior assistant response"))
        store.save_session(prior_session)

        # Resuming agent with prior session
        from terminal_agent.core.agent import Agent
        from terminal_agent.core.config import AgentConfig
        from terminal_agent.utils.approval import AutoApprovalHandler

        config = AgentConfig(provider="anthropic", anthropic_api_key="sk-test")
        agent = Agent(
            config=config,
            working_directory=temp_dir,
            session=prior_session,
            session_store=store,
            approval_handler=AutoApprovalHandler(approve=True),
        )

        assert agent.session.session_id == "resumed_123"
        # User message + assistant message are present
        convo = agent.session.get_conversation_messages()
        assert len(convo) == 2
        assert convo[0].content == "Prior user message"
        assert convo[1].content == "Prior assistant response"
