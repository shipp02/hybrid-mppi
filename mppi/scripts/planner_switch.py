"""Speed-scheduled choice between the MPPI command and the MPC tracker."""
import math

PLANNERS = ('mppi', 'mpc')


class PlannerSwitch:
    """Pick the planner whose command is published, with hysteresis.

    Starts on `low_speed`; moves to `high_speed` at or above `up_speed` and back
    at or below `down_speed`. A switch waits `min_dwell` ticks after the last
    one so speed noise near a threshold cannot make the planners chatter.
    """

    def __init__(self, low_speed, high_speed, up_speed, down_speed, min_dwell=0):
        if low_speed not in PLANNERS or high_speed not in PLANNERS:
            raise ValueError(f'planners must be one of {PLANNERS}')
        if down_speed > up_speed:
            raise ValueError('switch down speed must not exceed the up speed')
        if min_dwell < 0:
            raise ValueError('minimum dwell must be non-negative')
        self.low_speed, self.high_speed = low_speed, high_speed
        self.up_speed, self.down_speed, self.min_dwell = up_speed, down_speed, min_dwell
        self.reset()

    @classmethod
    def fixed(cls, planner):
        """Always the same planner."""
        return cls(planner, planner, math.inf, -math.inf)

    @property
    def uses(self):
        return {self.low_speed, self.high_speed}

    def reset(self):
        """Back to the low-speed planner, e.g. after a pose jump."""
        self.active = self.low_speed
        self._since_switch = self.min_dwell

    def select(self, speed):
        """Advance one control tick at `speed` and return the active planner."""
        self._since_switch += 1
        wanted = self._wanted(speed)
        if wanted != self.active and self._since_switch >= self.min_dwell:
            self.active, self._since_switch = wanted, 0
        return self.active

    def _wanted(self, speed):
        if speed >= self.up_speed:
            return self.high_speed
        if speed <= self.down_speed:
            return self.low_speed
        return self.active
