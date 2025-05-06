import subprocess
import sys
import time
import os
import psutil
import pytest

def test_simulate_large_scale(tmp_path):
    print("[TEST] Starting test_simulate_large_scale")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    cmd = [
        sys.executable, 'fl/run_multi_aggregator_simulation.py',
        '--scenario', 'single',
        '--clients', '12',
        '--rounds', '2',
        '--aggregators', '5',
        '--output-dir', str(output_dir)
    ]
    print(f"[TEST] Running subprocess: {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd)
    print("[TEST] Subprocess finished, checking output files...")
    files = list(output_dir.glob('research_data_*.json'))
    print(f"[TEST] Found output files: {files}")
    assert len(files) > 0
    print("[TEST] test_simulate_large_scale PASSED")


def test_performance_memory_and_runtime(tmp_path):
    print("[TEST] Starting test_performance_memory_and_runtime")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    start_time = time.time()
    cmd = [
        sys.executable, 'fl/run_multi_aggregator_simulation.py',
        '--scenario', 'single',
        '--clients', '5',
        '--rounds', '1',
        '--aggregators', '2',
        '--output-dir', str(output_dir)
    ]
    print(f"[TEST] Running subprocess: {' '.join(map(str, cmd))}")
    process = psutil.Popen(cmd)
    max_mem = 0
    while process.is_running():
        try:
            mem = process.memory_info().rss
            max_mem = max(max_mem, mem)
        except Exception:
            pass
        time.sleep(0.1)
    runtime = time.time() - start_time
    print(f"[TEST] Subprocess finished. Runtime: {runtime:.2f}s, Max memory: {max_mem/1e6:.2f}MB")
    assert runtime < 60
    assert max_mem < 1_000_000_000
    print("[TEST] test_performance_memory_and_runtime PASSED")


def test_configurable_experiment_scales(tmp_path):
    print("[TEST] Starting test_configurable_experiment_scales")
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    cmd = [
        sys.executable, 'fl/run_multi_aggregator_simulation.py',
        '--scenario', 'single',
        '--clients', '7',
        '--rounds', '3',
        '--aggregators', '4',
        '--output-dir', str(output_dir)
    ]
    print(f"[TEST] Running subprocess: {' '.join(map(str, cmd))}")
    subprocess.check_call(cmd)
    print("[TEST] Subprocess finished, checking output files...")
    files = list(output_dir.glob('research_data_*.json'))
    print(f"[TEST] Found output files: {files}")
    assert len(files) > 0
    with open(files[0]) as f:
        data = f.read()
    print(f"[TEST] Output file content: {data[:200]}...")
    assert 'total_aggregators' in data or 'total_rounds' in data
    print("[TEST] test_configurable_experiment_scales PASSED") 