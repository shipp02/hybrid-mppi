#!/usr/bin/env python3
"""Generate commit-ready tracker C/JSON; compilation belongs to colcon."""
import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'mppi/scripts'))

from acados_tracker import build_ocp
from controller_config import MPPIConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--horizon', type=int, default=MPPIConfig.H)
    parser.add_argument('--max-speed', type=float, default=MPPIConfig.MAX_SPEED)
    args = parser.parse_args()
    if args.horizon < 1 or not 0 < args.max_speed < float('inf'):
        parser.error('horizon and max-speed must be positive and finite')
    generate(replace(MPPIConfig(), H=args.horizon, MAX_SPEED=args.max_speed))


def generate(cfg):
    from acados_template import AcadosOcpSolver
    import numpy as np

    directory = ROOT / 'mppi/generated/acados_tracker'
    directory.mkdir(parents=True, exist_ok=True)
    ocp = build_ocp(cfg, np.pi, directory)
    ocp.code_gen_options.json_file = str(directory / 'acados_ocp.json')
    AcadosOcpSolver.generate(ocp)
    remove_obsolete_entrypoints(directory, ocp.name)
    make_formulation_portable(directory)
    normalize_generated_sources(directory)
    print(f'Generated sources: {directory}')
    print(f'Runtime configuration: horizon={cfg.H}, max_speed={cfg.MAX_SPEED}')
    print('Build with colcon --cmake-args -DMPPI_BUILD_ACADOS=ON')


def remove_obsolete_entrypoints(directory, problem_name):
    for pattern in ('acados_solver_*.c', 'acados_solver_*.h', 'main_*.c'):
        for path in directory.glob(pattern):
            if problem_name not in path.stem:
                path.unlink()


def make_formulation_portable(directory):
    path = directory / 'acados_ocp.json'
    payload = json.loads(path.read_text())
    options = payload['code_gen_options']
    options.update(acados_include_path='${ACADOS_SOURCE_DIR}/include',
                   acados_lib_path='${ACADOS_SOURCE_DIR}/lib',
                   code_export_directory='.', json_file='acados_ocp.json',
                   cython_include_dirs=[])
    path.write_text(json.dumps(payload, indent=4) + '\n')


def normalize_generated_sources(directory):
    """Remove upstream template whitespace so committed output passes diff checks."""
    for path in directory.rglob('*'):
        if path.suffix in ('.c', '.h'):
            lines = path.read_text().splitlines()
            path.write_text('\n'.join(line.rstrip() for line in lines) + '\n')


if __name__ == '__main__':
    main()
