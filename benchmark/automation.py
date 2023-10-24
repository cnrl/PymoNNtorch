import os
import sys
import platform
import subprocess
import pandas as pd
from tqdm import tqdm


def get_simulation_time(byte_out):
    str_out = byte_out.decode("utf-8")
    result = [
        line for line in str_out.split("\n") if line.startswith("simulation time:")
    ]
    return result


def get_number(line, script_loc):
    try:
        time = float(line.lstrip("simulation time:"))
    except RuntimeError:
        print(f"ERROR: couldn't get the time from string on script {script_loc}, line:")
        print(line)
        return None
    return time


def get_script_time(script_loc):
    result = get_simulation_time(subprocess.check_output(["python3", script_loc, 'no_plot']))
    if len(result) == 0:
        print(f"ERROR: no simulation time on {script_loc}")
    elif len(result) > 1:
        print(f"ERROR: multiple simulation time on {script_loc}")
    else:
        return get_number(result[0], script_loc)
    return None


if __name__ == "__main__":
    n = int(sys.argv[1])  # repeat
    out_loc = (
        f"{platform.node()}.csv" if sys.argv[2] == "_" else sys.argv[2]
    )  # csv output location
    scripts_loc = sys.argv[3:]  # list of script to run

    result = {}

    for _ in tqdm(range(n)):
        for script_loc in tqdm(scripts_loc):
            script_name = os.path.basename(script_loc)
            if script_name not in result:
                result[script_name] = []
            t = get_script_time(script_loc)
            if t is not None:
                result[script_name].append(t)
            # result[script_name].append(get_script_time(script_loc))

    new_df = pd.DataFrame(result)
    df = pd.DataFrame({})
    if os.path.exists(out_loc):
        df = pd.read_csv(out_loc, index_col=0)

    result_df = pd.concat([df, new_df], ignore_index=True, sort=False)
    result_df.to_csv(out_loc)
