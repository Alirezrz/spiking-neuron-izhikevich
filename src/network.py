from pymonntorch import *

class TimeResolution(Behavior):

    def initialize(self, network):
        super().initialize(network)
        network.dt = self.parameter("dt", 0.1)