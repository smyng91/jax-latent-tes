import argparse
import sys


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="pcm")
    p.add_argument("cmd", choices=["validate", "sweep", "explore", "baseline", "numbers", "figures"])
    p.add_argument("rest", nargs=argparse.REMAINDER)
    args = p.parse_args(argv)
    sys.argv = [sys.argv[0], *args.rest]
    if args.cmd == "validate":
        from pcm.validate import main as m
    elif args.cmd == "sweep":
        from pcm.sweep import main as m
    elif args.cmd == "explore":
        from pcm.explore import main as m
    elif args.cmd == "baseline":
        from pcm.baseline import main as m
    elif args.cmd == "numbers":
        from pcm.report import main_numbers as m
    else:
        from pcm.report import main_figures as m
    m()


if __name__ == "__main__":
    main()
