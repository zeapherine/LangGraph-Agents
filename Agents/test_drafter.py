import os
import tempfile
import pytest
from Agents.Drafter import update, save, AgentState

def test_update_tool():
    state = {"messages": [], "document_content": ""}
    updated_state = update.invoke({"state": state, "content": "Hello, Drafter!"})
    assert updated_state["document_content"] == "Hello, Drafter!"

# Test updating with empty content
def test_update_empty_content():
    state = {"messages": [], "document_content": "Old content"}
    updated_state = update.invoke({"state": state, "content": ""})
    assert updated_state["document_content"] == ""

def test_save_tool(tmp_path):
    state = {"messages": [], "document_content": "Test content for saving."}
    filename = tmp_path / "testfile.txt"
    result = save.invoke({"state": state, "filename": str(filename)})
    assert os.path.exists(filename)
    with open(filename, "r") as f:
        content = f.read()
    assert content == "Test content for saving."
    assert "saved successfully" in result.lower()

# Test saving with filename missing .txt extension
def test_save_tool_adds_txt(tmp_path):
    state = {"messages": [], "document_content": "Another test."}
    filename = tmp_path / "myfile"
    result = save.invoke({"state": state, "filename": str(filename)})
    assert os.path.exists(str(filename) + ".txt")
    with open(str(filename) + ".txt", "r") as f:
        content = f.read()
    assert content == "Another test."
    assert "saved successfully" in result.lower()

# Test saving with empty content
def test_save_empty_content(tmp_path):
    state = {"messages": [], "document_content": ""}
    filename = tmp_path / "empty.txt"
    result = save.invoke({"state": state, "filename": str(filename)})
    assert os.path.exists(filename)
    with open(filename, "r") as f:
        content = f.read()
    assert content == ""
    assert "saved successfully" in result.lower()

# Test saving with invalid path (should return error)
def test_save_invalid_path():
    state = {"messages": [], "document_content": "Should fail."}
    # Use an invalid path (directory does not exist)
    invalid_path = "/invalid_dir/should_fail.txt"
    result = save.invoke({"state": state, "filename": invalid_path})
    assert "error saving document" in result.lower()

if __name__ == "__main__":
    pytest.main([__file__])
