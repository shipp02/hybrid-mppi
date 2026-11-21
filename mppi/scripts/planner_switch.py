"""Speed-scheduled choice between the MPPI command and the MPC tracker."""
import math

PLANNERS = ('mppi', 'mpc')


class PlannerSwitch:
    """Pick the planner whose command is published, with hysteresis.

    Starts on `low_planner`. The high-speed band is entered at or above
    `high_enter` (switch to `high_planner`) and left at or below `high_exit`
    (back to `low_planner`). A switch waits `min_dwell` ticks after the last
    one so speed noise near a threshold cannot make the planners chatter.
    """

    def __init__(self, low_planner, high_planner, high_enter, high_exit, min_dwell=0):
        if low_planner not in PLANNERS or high_planner not in PLANNERS:
            raise ValueError(f'planners must be one of {PLANNERS}')
        if high_exit >= high_enter:
            raise ValueError('high_exit speed must be below high_enter speed')
        if min_dwell < 0:
            raise ValueError('minimum dwell must be non-negative')
        self.low_planner, self.high_planner = low_planner, high_planner
        self.high_enter, self.high_exit, self.min_dwell = high_enter, high_exit, min_dwell
        self.reset()

    @classmethod
    def fixed(cls, planner):
        """Always the same planner."""
        return cls(planner, planner, math.inf, -math.inf)

    @property
    def uses(self):
        return {self.low_planner, self.high_planner}

    def reset(self):
        """Back to the low-speed planner, e.g. after a pose jump."""
        self.active = self.low_planner
        self._since_switch = self.min_dwell

    def select(self, speed):
        """Advance one control tick at `speed` and return the active planner."""
        self._since_switch += 1
        wanted = self._wanted(speed)
        if wanted != self.active and self._since_switch >= self.min_dwell:
            self.active, self._since_switch = wanted, 0
        return self.active

    def _wanted(self, speed):
        if speed >= self.high_enter:
            return self.high_planner
        if speed <= self.high_exit:
            return self.low_planner
        return self.active
