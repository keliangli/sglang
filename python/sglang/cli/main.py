import argparse

from sglang.cli.generate import generate
from sglang.cli.gen_throughput_params import generate_throughput_params
from sglang.cli.serve import serve


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    serve_parser = subparsers.add_parser(
        "serve",
        help="Launch the SGLang server.",
        add_help=False,  # Defer help to the specific parser
    )
    serve_parser.set_defaults(func=serve)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Run inference on a multimodal model.",
        add_help=False,  # Defer help to the specific parser
    )
    generate_parser.set_defaults(func=generate)

    gen_params_parser = subparsers.add_parser(
        "gen-throughput-params",
        help="Generate throughput optimization parameter combinations.",
        add_help=False,  # Defer help to the specific parser
    )
    gen_params_parser.set_defaults(func=generate_throughput_params)

    args, extra_argv = parser.parse_known_args()
    args.func(args, extra_argv)
