import time 
import numpy as np


class TrafficLight:
    """TrafficLight Class"""

    def __init__(self, position, state='green'):

        assert state in ['green', 'yellow', 'red'], "Invalid traffic light state"
        
        self.position = position
        self.state = state  # 'green', 'yellow', 'red'
        self.min_green_time = 10 # Minimum time the light stays green to avoid rapid switching

    def valid_switch():
        """Check if the traffic light can switch state."""
        pass

    def switch_to(self, delta_time = 0, switch_time_countdown = 0):
        """Switch the traffic light state based on a simple timer."""
        
        assert self.valid_switch(), "Traffic light cannot switch state now."

        # Implement the delta time between the light switching
        time.sleep(delta_time + switch_time_countdown)
        if self.state == 'green':
            self.state = 'yellow'
        elif self.state == 'yellow':
            self.state = 'red'
        elif self.state == 'red':
            self.state = 'green'

    def get_info(self):
        """Return the info of traffic light at the current time step."""
        return self.state, self.position, self.min_green_time, self.valid_switch()
