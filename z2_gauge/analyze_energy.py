#!/usr/bin/env python3
"""Energy cumulants per lattice site, with blocked jackknife errors."""

import argparse
import gzip
from itertools import chain, islice

import numpy as np


def read_blocks(filename, block_size, discard, reference):
    """Read one replica and keep only the first four power sums per block."""
    open_file = gzip.open if filename.endswith(".gz") else open
    with open_file(filename, "rt", encoding="utf-8") as file:
        header = {}
        for line in file:
            line = line.strip()
            if line.startswith("#"):
                key, separator, value = line[1:].partition("=")
                if separator:
                    header[key.strip()] = value.strip()
            elif line:
                break
        else:
            raise ValueError("no energy measurements")

        L = int(header["L"])
        beta = float(header["beta"])
        measure_every = int(header["measure_every"])
        seed = int(header["seed"])
        if L < 2 or not np.isfinite(beta) or beta < 0 or measure_every < 1:
            raise ValueError("invalid L, beta, or measure_every in the header")

        rows = (row for row in chain((line,), file)
                if row.strip() and not row.lstrip().startswith("#"))
        rows = islice(rows, discard, None)
        # Align chunks with blocks. If a block is large, read one at a time.
        chunk_rows = max(1, 100_000 // block_size) * block_size
        blocks = []
        tail = 0
        columns = None
        while lines := list(islice(rows, chunk_rows)):
            data = np.loadtxt(lines, ndmin=2)
            del lines
            if columns is None:
                columns = data.shape[1]
            if columns not in (1, 3) or data.shape[1] != columns:
                raise ValueError("expected total_energy or sweep total_energy acceptance")
            energy = data[:, 0 if columns == 1 else 1]
            if not np.all((np.abs(energy) <= 3 * L**3) & (energy == np.floor(energy))):
                raise ValueError("invalid total energy")

            tail = len(energy) % block_size
            used = len(energy) - tail
            if used == 0:
                break
            if reference is None:
                # Any common offset works. A nearby one limits cancellation.
                reference = energy[:used].mean()
            shifted = (energy[:used] - reference).reshape(-1, block_size)
            squared = shifted**2
            blocks.append(np.column_stack((
                shifted.sum(axis=1),
                squared.sum(axis=1),
                (squared * shifted).sum(axis=1),
                (squared**2).sum(axis=1),
            )))

    if not blocks:
        raise ValueError("no complete blocks after discarding measurements")
    return (L, beta, measure_every), seed, reference, blocks, tail


def cumulants(moments, volume):
    """Convert moments about the common reference to cumulants of E."""
    m1, m2, m3, m4 = moments.T
    variance = m2 - m1**2
    central3 = m3 - 3 * m1 * m2 + 2 * m1**3
    central4 = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4
    return np.stack((central3, central4 - 3 * variance**2), axis=-1) / volume


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", help="replicas at the same L and beta (.dat or .dat.gz)")
    parser.add_argument("--block-size", type=int, required=True, help="measurements per block")
    parser.add_argument("--discard", type=int, default=0, help="initial measurements to skip in each file")
    parser.add_argument("--blocks-out", help="optional text file for the four power sums per block")
    args = parser.parse_args()
    if args.block_size < 1 or args.discard < 0:
        parser.error("block size must be positive and discard must be nonnegative")

    settings = None
    reference = None
    seeds = set()
    block_chunks = []
    dropped = 0
    for filename in args.files:
        try:
            current, seed, reference, blocks, tail = read_blocks(
                filename, args.block_size, args.discard, reference
            )
            if settings is not None and current != settings:
                raise ValueError("replicas must have matching L, beta, and measure_every")
            if seed in seeds:
                raise ValueError("replicas must have distinct seeds")
        except (OSError, EOFError, ValueError, KeyError) as error:
            parser.error(f"{filename}: {error}")
        settings = current
        seeds.add(seed)
        block_chunks.extend(blocks)
        dropped += tail

    L, beta, measure_every = settings
    block_sums = np.concatenate(block_chunks)
    nblocks = len(block_sums)
    if nblocks < 2:
        parser.error("need at least two complete blocks for the jackknife")

    # Evaluate the cumulants on all samples and again with each block removed.
    used = nblocks * args.block_size
    total = block_sums.sum(axis=0)
    values = cumulants(total / used, L**3)
    leave_one_out = cumulants((total - block_sums) / (used - args.block_size), L**3)
    deviations = leave_one_out - leave_one_out.mean(axis=0)
    errors = np.sqrt((nblocks - 1) / nblocks * (deviations**2).sum(axis=0))

    if args.blocks_out:
        header = (f"L = {L}\nbeta = {beta:.17g}\nmeasure_every = {measure_every}\n"
                  f"block_size = {args.block_size}\nenergy_reference = {reference:.17g}\n"
                  "columns: sum(E-ref) sum((E-ref)^2) sum((E-ref)^3) sum((E-ref)^4)")
        try:
            with open(args.blocks_out, "x", encoding="utf-8") as output:
                np.savetxt(output, block_sums, fmt="%.17g", header=header)
        except OSError as error:
            parser.error(str(error))

    print(f"L = {L}, beta = {beta:.12g}")
    print(f"Used {used} measurements in {nblocks} blocks from {len(args.files)} file(s).")
    print(f"Skipped {args.discard * len(args.files)} initial measurements and "
          f"dropped {dropped} from incomplete blocks.")
    for name, value, error in zip(("C3", "C4"), values, errors):
        print(f"{name} = {value:.10g} +/- {error:.10g}")


if __name__ == "__main__":
    main()
