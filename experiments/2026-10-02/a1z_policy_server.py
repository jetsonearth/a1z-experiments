#!/usr/bin/env python3
"""Project-local ACT server entry point; does not import or enable the robot SDK."""
from datetime import datetime
from pathlib import Path

from lerobot.async_inference import policy_server as upstream

from async_policy import FreshObservationServerMixin, Timeline


class A1ZPolicyServer(FreshObservationServerMixin, upstream.PolicyServer):
    def __init__(self, config):
        super().__init__(config)
        stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
        path = Path(__file__).resolve().parent/'server-runs'/f'{stamp}-async.jsonl'
        self.async_trace = Timeline(path)
        self.async_trace.record('server_start', observation_filter='fresh_timestep_v1')
        self.async_trace.flush()
        self.logger.info('A1Z fresh-observation server; timing trace: %s', path)


if __name__ == '__main__':
    # Reuse upstream CLI and transport without changing the installed package.
    upstream.PolicyServer = A1ZPolicyServer
    upstream.serve()
