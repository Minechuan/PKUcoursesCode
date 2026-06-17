from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_questions
from .experiment import build_chunks, run_experiment


def parse_float_list(value: str) -> list[float]:
    return [float(v.strip()) for v in value.split(",") if v.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FinanceBench BM25+dense hybrid retrieval experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--questions", default="financebench-main/data/financebench_open_source.jsonl")
    common.add_argument("--pdf-dir", default="financebench-main/pdfs")
    common.add_argument("--cache-dir", default="code/artifacts")
    common.add_argument("--subset", default="OPEN_SOURCE")
    common.add_argument("--chunk-size", type=int, default=1200)
    common.add_argument("--chunk-overlap", type=int, default=200)
    common.add_argument("--force-extract", action="store_true")

    prep = sub.add_parser("prepare-corpus", parents=[common], help="extract PDF pages and build chunks")
    prep.set_defaults(func=cmd_prepare)

    run = sub.add_parser("run", parents=[common], help="run retrieval and fusion ablations")
    run.add_argument("--output-dir", default="code/outputs")
    run.add_argument("--embedding-model", default="BAAI/bge-base-en-v1.5")
    run.add_argument("--weights", type=parse_float_list, default=parse_float_list("0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1"))
    run.add_argument("--fusion", choices=["linear", "rrf"], default="linear")
    run.add_argument("--top-k", type=parse_int_list, default=parse_int_list("5,10"))
    run.add_argument("--retrieval-pool", type=int, default=50)
    run.add_argument("--dense-batch-size", type=int, default=32)
    run.set_defaults(func=cmd_run)
    return parser


def cmd_prepare(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions, subset=args.subset)
    _, chunks = build_chunks(
        questions_path=args.questions,
        pdf_dir=args.pdf_dir,
        cache_dir=args.cache_dir,
        subset=args.subset,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        force_extract=args.force_extract,
    )
    print(f"Loaded {len(questions)} questions")
    print(f"Prepared {len(chunks)} chunks")
    print(f"Page cache: {Path(args.cache_dir) / 'pages.jsonl'}")


def cmd_run(args: argparse.Namespace) -> None:
    metrics = run_experiment(
        questions_path=args.questions,
        pdf_dir=args.pdf_dir,
        cache_dir=args.cache_dir,
        output_dir=args.output_dir,
        subset=args.subset,
        embedding_model=args.embedding_model,
        weights=args.weights,
        fusion=args.fusion,
        top_ks=args.top_k,
        retrieval_pool=args.retrieval_pool,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dense_batch_size=args.dense_batch_size,
        force_extract=args.force_extract,
    )
    print(metrics.to_string(index=False))
    print(f"Saved metrics to {Path(args.output_dir) / 'metrics.csv'}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
