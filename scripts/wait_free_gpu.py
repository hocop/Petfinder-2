import argparse
import os
import subprocess
import time


def main(args):
    minutes = 0

    while True:
        cmd = f'nvidia-smi -i {args.gpu} | grep % | cut -c 36-55'
        mem = subprocess.check_output(cmd, shell=True)
        mem = mem.decode("utf-8").strip().split()[0].replace('MiB', '')
        mem = int(mem)
        print('Waiting', minutes, 'minutes.', 'Used:', mem, 'MB')

        if mem < args.threshold:
            break

        time.sleep(60)
        minutes += 1


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
