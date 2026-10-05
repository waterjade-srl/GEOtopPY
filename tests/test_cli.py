"""Check that the public command runs the verified simulation pipeline."""

from geotop_py import cli, pipeline


def test_cli_runs_the_verified_pipeline():
    assert cli.run_simulation is pipeline.run_simulation
