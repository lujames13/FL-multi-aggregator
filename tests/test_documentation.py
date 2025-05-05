import os
import subprocess
import sys
import re

import pytest

def test_all_cli_parameters_documented():
    """Test that all CLI parameters are documented in the help output."""
    # Check analyze_results.py
    help_out = subprocess.check_output([sys.executable, 'fl/analyze_results.py', '--help'], text=True)
    assert '--results-dir' in help_out
    assert '--output-dir' in help_out
    # Check run_multi_aggregator_simulation.py
    help_out2 = subprocess.check_output([sys.executable, 'fl/run_multi_aggregator_simulation.py', '--help'], text=True)
    assert '--scenario' in help_out2
    assert '--clients' in help_out2
    assert '--rounds' in help_out2
    assert '--aggregators' in help_out2
    assert '--malicious' in help_out2
    assert '--challenges' in help_out2
    assert '--output-dir' in help_out2
    assert '--visualize' in help_out2


def test_example_research_workflow(tmp_path):
    """Test that an example research workflow can be run and produces expected results files."""
    # Simulate running the analysis script on a dummy results dir
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    # Create a dummy research_data file
    (results_dir / "research_data_example.json").write_text('{"total_rounds": 3, "challenge_success_rate": 1.0}')
    # Run analysis
    subprocess.check_call([sys.executable, 'fl/analyze_results.py', '--results-dir', str(results_dir)])
    # Check that output files are created
    vis_dir = results_dir / "visualizations"
    assert (vis_dir / "challenge_effectiveness.png").exists()
    assert (vis_dir / "aggregator_performance.png").exists()
    assert (vis_dir / "challenge_timeline.png").exists()
    assert (vis_dir / "comparative_analysis.json").exists()
    assert (vis_dir / "research_report.txt").exists()


def test_documentation_is_comprehensive_and_up_to_date():
    """Test that the README documents all major features and usage."""
    with open('README.md') as f:
        readme = f.read()
    # Check for key sections
    assert 'Installation' in readme
    assert 'Usage' in readme
    assert 'Advanced Configuration' in readme
    assert 'Research Applications' in readme
    # Check for mention of challenge mechanism and malicious strategies
    assert 'challenge mechanism' in readme.lower()
    assert 'malicious' in readme.lower()
    assert 'visualization' in readme.lower()


def test_limitations_and_poc_scope_are_stated():
    """Test that limitations and PoC scope are clearly stated in the documentation."""
    with open('README.md') as f:
        readme = f.read()
    # Look for limitations or PoC scope
    assert re.search(r'limitation|proof[- ]of[- ]concept|poc', readme, re.IGNORECASE) 