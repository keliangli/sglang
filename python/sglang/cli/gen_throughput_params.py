"""CLI entry point for throughput parameter combination generator."""

import sys

from sglang.throughput_param_generator import main


def generate_throughput_params(args, extra_argv):
    """Entry point for sglang gen-throughput-params command."""
    sys.argv = ["sglang gen-throughput-params"] + extra_argv
    return main()
