import argparse
from pathlib import Path
import subprocess


scripts_dir = Path(__file__).resolve().parent
decode_advi_py = scripts_dir / "decode_advi.py"
assert decode_advi_py.exists()
decode_cavi_py = scripts_dir / "decode_cavi.py"
assert decode_cavi_py.exists()
h5_to_numpy_py = scripts_dir / "h5_to_numpy.py"
assert h5_to_numpy_py.exists()


grep_prefix = "[decode_re]:"


def decode(
    pid,
    ephys_path,
    out_path,
    roi,
    which="cavi",
    batch_size=None,
    max_iter=None,
    learning_rate=None,
    behavior=None,
):
    if which == "cavi":
        decode_py = decode_cavi_py
    elif which == "advi":
        decode_py = decode_advi_py
    else:
        assert False
    extra = []
    if batch_size is not None:
        extra.append(f"--batch_size={batch_size}")
    if max_iter is not None:
        extra.append(f"--max_iter={max_iter}")
    if learning_rate is not None:
        extra.append(f"--learning_rate={learning_rate}")
    if behavior is not None:
        extra.append(f"--behavior={behavior}")
    subprocess.run(
        [
            "python",
            decode_py,
            f"--pid={pid}",
            f"--ephys_path={ephys_path}",
            f"--out_path={out_path}",
            f"--brain_region={roi}",
            "--featurize_behavior",
            *extra,
        ]
    )


def process_pid(pid, ephys_path, out_path):
    subprocess.run(["python", h5_to_numpy_py, f"--root_path={ephys_path}"])
    
    print(grep_prefix, "Decoding binary choices")
    for roi in ["ca1", "dg", "lp", "po", "visa"]:
        decode(pid, ephys_path, out_path, roi, max_iter=20)
    decode(
        pid,
        ephys_path,
        out_path,
        "all",
        batch_size=1,
        max_iter=20,
        learning_rate="1e-2",
    )

    print(grep_prefix, "Decoding wheel speed")
    for roi in ["ca1", "dg", "lp", "po", "visa", "all"]:
        decode(
            pid,
            ephys_path,
            out_path,
            roi,
            which="advi",
            behavior="wheel_speed",
            batch_size=1,
            learning_rate="1e-2",
        )

    print(grep_prefix, "Decoding motion energy")
    for roi in ["ca1", "dg", "lp", "po", "visa", "all"]:
        decode(
            pid,
            ephys_path,
            out_path,
            roi,
            which="advi",
            behavior="motion_energy",
            batch_size=1,
            learning_rate="1e-2",
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()

    ap.add_argument("ephys_base_path", type=Path)
    ap.add_argument("out_path", type=Path)
    ap.add_argument("--pids", type=str, required=True)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    # PIDs are input separated by commas or in a file
    if Path(args.pids).exists():
        with open(args.pids, "r") as pidfile:
            args.pids = pidfile.read().strip().split()
    else:
        args.pids = args.pids.split(",")
    print(grep_prefix, "Will run on these pids:\n", grep_prefix, "- ", f"\n{grep_prefix} - ".join(args.pids))

    # create outdir
    args.out_path.mkdir(exist_ok=True)
    print(grep_prefix, f"Saving to {args.out_path}")

    # run the loop
    for pid in args.pids:
        ephys_path = args.ephys_base_path / pid
        print(grep_prefix, pid)

        if not ephys_path.exists():
            print(f"{grep_prefix} No ephys dir for {pid=}. Skip.")
            continue
        
        if (args.out_path / pid).exists() and not args.overwrite:
            print(f"{grep_prefix} {pid=} output dir exists and {args.overwrite=}, moving on.")
            continue

        process_pid(pid, ephys_path, args.out_path)
