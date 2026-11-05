import argparse

from .bot import FAQBot, load, save
from .generate import generate_folder


def main() -> None:
    ap = argparse.ArgumentParser(prog="faqforge")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate faq.json from a docs folder"); b.add_argument("docs"); b.add_argument("-o", "--out", default="faq.json")
    a = sub.add_parser("ask"); a.add_argument("faq"); a.add_argument("question")
    c = sub.add_parser("chat"); c.add_argument("faq"); c.add_argument("--logs", default=".")
    g = sub.add_parser("gaps", help="top unanswered questions"); g.add_argument("faq"); g.add_argument("--logs", default=".")
    args = ap.parse_args()
    if args.cmd == "build":
        pairs = generate_folder(args.docs); save(pairs, args.out)
        for p in pairs:
            print(f"{p.id:2}. {p.question}")
        print(f"-> {len(pairs)} Q&A pairs written to {args.out}")
    elif args.cmd == "ask":
        r = FAQBot(load(args.faq)).ask(args.question)
        print(f"[{r.kind} {r.confidence}] {r.text}")
        for s in r.suggestions:
            print("  -", s.question)
    elif args.cmd == "gaps":
        for q, n in FAQBot(load(args.faq), log_dir=args.logs).gaps():
            print(f"{n:3}x {q}")
    else:
        bot = FAQBot(load(args.faq), log_dir=args.logs)
        while True:
            try:
                q = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            r = bot.ask(q)
            print(f"bot[{r.kind}]> {r.text}")
            for s in r.suggestions:
                print("   -", s.question)


main()
