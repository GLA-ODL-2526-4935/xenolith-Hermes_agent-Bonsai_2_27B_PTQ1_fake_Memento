"""Owner-only administration. Never register these commands as model tools."""
import argparse
import json

from .broker import Broker
from .service import serve


def initialize(args):
    _broker, token = Broker.create(args.root, scope=args.scope)
    return {"client_token": token, "notice": "Shown once. Store as a secret; never in model context."}


def inspect_request(args):
    return Broker(args.root).inspect(args.id)


def list_requests(args):
    return Broker(args.root).list_pending()


def decide(args):
    return Broker(args.root).decide(args.id, args.digest, approve=args.action == "approve")


def run_service(args):
    serve(Broker(args.root), args.port)
    return {"status": "stopped"}


COMMANDS = {"init": initialize, "inspect": inspect_request, "list": list_requests,
            "approve": decide, "reject": decide, "serve": run_service}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="Dedicated owner-side broker directory, never mounted in agent")
    commands = parser.add_subparsers(dest="action", required=True)
    for name, handler in COMMANDS.items():
        child = commands.add_parser(name)
        child.set_defaults(handler=handler)
        if name == "init":
            child.add_argument("--scope", choices=("manual", "research"), default="manual",
                               help="Research refuses exports/downloads/uploads/forms/messages/account changes")
        if name in {"inspect", "approve", "reject"}:
            child.add_argument("id")
        if name in {"approve", "reject"}:
            child.add_argument("digest", help="Exact digest reviewed with inspect")
        if name == "serve":
            child.add_argument("--port", type=int, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(args.handler(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
