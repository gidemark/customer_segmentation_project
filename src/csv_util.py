import os
import shutil


def write_single_csv(df, path):
    """Write a Spark DataFrame as a single flat CSV file at `path`.

    Spark's CSV writer always emits a directory. This helper writes to a temp
    sibling directory, moves the single part file to `path`, then cleans up.
    """
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        shutil.rmtree(tmp, ignore_errors=True)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp)
    part_files = [f for f in os.listdir(tmp) if f.startswith("part-")]
    if not part_files:
        raise RuntimeError(f"No part file produced when writing {path}")
    src = os.path.join(tmp, part_files[0])
    if os.path.exists(path):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        else:
            os.remove(path)
    shutil.move(src, path)
    shutil.rmtree(tmp, ignore_errors=True)
    return path