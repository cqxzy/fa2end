#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

AA20 = set("ACDEFGHIKLMNPQRSTVWY")

# 固定为你已经跑通的 AF3 输入格式
DEFAULT_DIALECT = "alphafold3"
DEFAULT_VERSION = 1
DEFAULT_SEEDS = [1]
DEFAULT_CHAIN_ID = "A"


def read_fasta(path: Path) -> List[Tuple[str, str]]:
    records: List[Tuple[str, str]] = []
    header: Optional[str] = None
    buf: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(buf)))
            header = line[1:].strip()
            buf = []
        else:
            buf.append(line)
    if header is not None:
        records.append((header, "".join(buf)))
    return records


def clean_seq(seq: str) -> str:
    seq = re.sub(r"\s+", "", seq).upper()
    bad = set(seq) - AA20
    if bad:
        raise ValueError(f"Invalid amino-acid letters found: {sorted(bad)}")
    return seq


def sanitize_name(s: str, max_len: int = 160) -> str:
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return (s or "job")[:max_len]


def parse_seeds(seeds_str: Optional[str]) -> List[int]:
    if not seeds_str:
        return DEFAULT_SEEDS
    seeds = [int(x) for x in seeds_str.split(",") if x.strip()]
    if not seeds:
        raise ValueError("--seeds parsed empty; example: --seeds 1 or --seeds 1,2")
    return seeds


def collect_fasta_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    if input_path.is_dir():
        fasta_files = sorted(
            (p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in {".fa", ".fasta"}),
            key=lambda p: p.name.lower(),
        )
        if not fasta_files:
            raise ValueError(f"No FASTA files found in directory: {input_path} (expect .fa/.fasta)")
        return fasta_files
    raise ValueError(f"--input path does not exist or is not a file/directory: {input_path}")


def write_json_jobs(
    fasta_path: Path,
    json_dir: Path,
    prefix: str,
    seeds: List[int],
    max_n: int,
    skip_existing: bool,
    chain_delim: Optional[str],
    target_seq: Optional[str],
    target_id: str,
    design_id: str,
) -> int:
    records = read_fasta(fasta_path)
    if max_n and max_n > 0:
        records = records[:max_n]

    json_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for i, (_hdr, seq_raw) in enumerate(records, start=1):
        name = sanitize_name(f"{prefix}_{i:04d}")
        out_path = json_dir / f"{name}.json"
        if skip_existing and out_path.exists():
            continue

        seq_raw = re.sub(r"\s+", "", seq_raw).upper()
        sequences: List[Dict[str, Any]] = []

        if target_seq is not None:
            # 2-chain: fixed target + designed chain per record
            sequences = [
                {"protein": {"id": [target_id], "sequence": target_seq}},
                {"protein": {"id": [design_id], "sequence": clean_seq(seq_raw)}},
            ]
        else:
            # single-chain (default) OR split by delimiter
            if chain_delim and chain_delim in seq_raw:
                parts = [p for p in seq_raw.split(chain_delim) if p]
                for k, part in enumerate(parts):
                    cid = chr(ord("A") + k)
                    sequences.append({"protein": {"id": [cid], "sequence": clean_seq(part)}})
            else:
                sequences = [{"protein": {"id": [DEFAULT_CHAIN_ID], "sequence": clean_seq(seq_raw)}}]

        job = {
            "name": name,
            "sequences": sequences,
            "modelSeeds": seeds,
            "dialect": DEFAULT_DIALECT,
            "version": DEFAULT_VERSION,
        }

        out_path.write_text(json.dumps(job, indent=2))
        written += 1

    return written

def run_af3_docker(
    json_dir: Path,
    af_output: Path,
    image: str,
    models_dir: Path,
    databases_dir: Path,
    gpus: str,
    interactive: bool,
):
# def run_af3_docker(
#     json_dir: Path,
#     af_output: Path,
#     image: str,
#     models_dir: Path,
#     databases_dir: Path,
#     gpus: str,
#     interactive: bool,
#     model_preset: Optional[str],

# ):
    # 关键：不要求 json_dir 必须在某个 inputs/ 下
    # 直接把 json_dir 挂载成容器里的 /root/af_input
    json_dir.mkdir(parents=True, exist_ok=True)
    af_output.mkdir(parents=True, exist_ok=True)

    cmd = ["docker", "run", "--rm"]
    if interactive:
        cmd.append("-it")
    cmd += ["--gpus", gpus]
    cmd += ["--volume", f"{json_dir.resolve()}:/root/af_input"]
    cmd += ["--volume", f"{af_output.resolve()}:/root/af_output"]
    cmd += ["--volume", f"{models_dir.resolve()}:/root/models"]
    cmd += ["--volume", f"{databases_dir.resolve()}:/root/public_databases"]
    cmd += [
        image, "python", "run_alphafold.py",
        "--input_dir=/root/af_input",
        "--model_dir=/root/models",
        "--output_dir=/root/af_output"
    ]
    # cmd += [
    #     image, "python", "run_alphafold.py",
    #     "--input_dir=/root/af_input",
    #     "--model_dir=/root/models",
    #     "--output_dir=/root/af_output"
    # ]

    # if model_preset:
    #     model_preset = model_preset.strip()
    #     if model_preset:

    #         cmd.append(f"--model_preset={model_preset}")


    print("\n[AF3] Running docker command:\n" + " ".join(cmd) + "\n")
    subprocess.run(cmd, check=True)


def main():
    ap = argparse.ArgumentParser(
        description="One-command AF3 pipeline: FASTA -> JSONs (fixed alphafold3/version=1), optionally run AF3 docker."
    )
    ap.add_argument("--input", required=True, help="Input FASTA file or directory (.fa/.fasta).")
    ap.add_argument("--json-dir", required=True, help="Where to write JSON files (any path you like).")
    ap.add_argument("--prefix", default=None, help="Job name prefix (default: FASTA stem).")
    ap.add_argument("--seeds", default=None, help="Comma seeds, e.g. '1' or '1,2'. Default: 1.")
    ap.add_argument("--max-n", type=int, default=0, help="0 means all records; otherwise only first N.")
    ap.add_argument("--skip-existing", action="store_true", help="Skip JSON if already exists.")
    ap.add_argument("--chain-delim", default=None, help="Optional delimiter to split one record into chains, e.g. '/'.")

    # binder mode (optional)
    ap.add_argument("--target", default=None, help="Target FASTA (exactly 1 sequence) for 2-chain jobs.")
    ap.add_argument("--target-id", default="A")
    ap.add_argument("--design-id", default="B")

    # run AF3 (optional)
    ap.add_argument("--run", action="store_true", help="Run AlphaFold3 docker after generating JSONs.")
    ap.add_argument("--af-output", default=None, help="AF3 output directory (required if --run).")
    ap.add_argument("--image", default="alphafold3:latest")
    ap.add_argument("--models", default="/data/models")
    ap.add_argument("--databases", default="/data/public_databases")
    ap.add_argument("--gpus", default="all", help="Docker GPU spec: all or device=1")
    ap.add_argument("--interactive", action="store_true", help="Add -it to docker run (like admin example).")
    # ap.add_argument("--af-output", default=None, help="AF3 output directory (required if --run).")
    # ap.add_argument("--image", default="alphafold3:latest")
    # ap.add_argument("--models", default="/data/models")
    # ap.add_argument("--databases", default="/data/public_databases")
    # ap.add_argument("--model_preset", default=None, help="Model preset passed to run_alphafold.py, e.g. multimer.")

    # ap.add_argument("--gpus", default="all", help="Docker GPU spec: all or device=1")
    # ap.add_argument("--interactive", action="store_true", help="Add -it to docker run (like admin example).")

    args = ap.parse_args()

    input_path = Path(args.input).resolve()
    fasta_files = collect_fasta_inputs(input_path)
    input_is_single_file = input_path.is_file()
    json_dir = Path(args.json_dir).resolve()
    seeds = parse_seeds(args.seeds)
    base_prefix = args.prefix or input_path.stem

    target_seq = None
    if args.target:
        tpath = Path(args.target).resolve()
        trecs = read_fasta(tpath)
        if len(trecs) != 1:
            raise ValueError(f"--target FASTA must contain exactly 1 sequence, got {len(trecs)}")
        target_seq = clean_seq(trecs[0][1])

    total_written = 0
    processed_files = 0
    for fasta_file in fasta_files:
        if args.max_n and args.max_n > 0:
            remaining = args.max_n - total_written
            if remaining <= 0:
                break
        else:
            remaining = 0

        if input_is_single_file:
            file_prefix = base_prefix
        else:
            file_prefix = sanitize_name(f"{base_prefix}_{fasta_file.name}")

        n = write_json_jobs(
            fasta_path=fasta_file,
            json_dir=json_dir,
            prefix=file_prefix,
            seeds=seeds,
            max_n=remaining,
            skip_existing=args.skip_existing,
            chain_delim=args.chain_delim,
            target_seq=target_seq,
            target_id=args.target_id,
            design_id=args.design_id,
        )
        total_written += n
        processed_files += 1

    print(f"[OK] Processed {processed_files} FASTA files, wrote {total_written} JSON files to: {json_dir}")
    print(f"Fixed JSON format: dialect={DEFAULT_DIALECT}, version={DEFAULT_VERSION}, modelSeeds={seeds}")

    if args.run:
        if not args.af_output:
            raise SystemExit("ERROR: --af-output is required when using --run")
        run_af3_docker(
            json_dir=json_dir,
            af_output=Path(args.af_output).resolve(),
            image=args.image,
            models_dir=Path(args.models),
            databases_dir=Path(args.databases),
            gpus=args.gpus,
            interactive=args.interactive,
            # models_dir=Path(args.models),
            # databases_dir=Path(args.databases),
            # gpus=args.gpus,
            # interactive=args.interactive,
            # model_preset=args.model_preset,

        )


if __name__ == "__main__":
    main()
