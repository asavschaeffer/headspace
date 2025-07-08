"""Test script for the Glass Engine instrumentation."""

import os
from click.testing import CliRunner
from globule.cli import cli

def test_add_instrumentation():
    """Test that the add command creates a trace."""
    if os.path.exists("trace.log"):
        os.remove("trace.log")

    runner = CliRunner()
    result = runner.invoke(cli, ["add", "My first instrumented thought"], env={"GLASS_ENGINE_ENABLED": "true"})
    
    assert result.exit_code == 0

    with open("trace.log", "r") as f:
        trace_content = f.read()
        assert "process_thought" in trace_content

if __name__ == "__main__":
    test_add_instrumentation()