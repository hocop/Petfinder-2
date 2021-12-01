import argparse
import os
import subprocess
import time


def get_mem():
    cmd = f'nvidia-smi -i {args.gpu} | grep % | cut -c 36-55'
    mem = subprocess.check_output(cmd, shell=True)
    mem = mem.decode("utf-8").strip().split()[0].replace('MiB', '')
    return int(mem)

def main(args):
    minutes = 0

    if get_mem() < args.threshold:
        print('GPU is free')
        return
    print('Waiting. Used:', get_mem(), 'MB')

    while True:
        minutes += 1

        mems = []
        for i in range(60):
            mems.append(get_mem())
            time.sleep(1)

        print('Waiting', minutes, 'minute(s).', 'Used min:', min(mems), 'MB, max:', max(mems), 'MB')

        if max(mems) < args.threshold:
            break


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Sleep until selected GPU is free')

    parser.add_argument(
        'gpu', type=int,
        help='Index of GPU')
    parser.add_argument(
        '--threshold', type=int, default=200,
        help='Maximum used memory, when GPU is considered free. In MB.')

    args = parser.parse_args()

    # Run program
    main(args)
