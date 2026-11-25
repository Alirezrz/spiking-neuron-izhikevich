from pymonntorch import *




# ConstantCurrent : I = I_value
# StepCurrent     : I = 0 before t_start, I = I_value after
# NoisyCurrent    : I = I_value + noise_std * N(0, 1)



class ConstantCurrent( Behavior):
    def initialize(self, neurons):
        self.I=self.parameter("I",10)
        neurons.I=neurons.vector(mode=self.I)
    
    def forward(self,neurons):
        neurons.I=neurons.vector(mode=self.I)
            
            
            
            
class StepCurrent(Behavior):
    
    def initialize(self, neurons):
        self.I=self.parameter("I",10.0)
        
        self.start=self.parameter("t_start",0.0)
        
        neurons.I=neurons.vector(mode=0.0)
        
        
        
    def forward(self,neurons):
        t = neurons.network.iteration *neurons.network.dt
        neurons.I=neurons.vector(mode=self.I if t >= self.start else 0.0)    
        
        

class NoisyCurrent(Behavior):

    def initialize(self, neurons):
        self.I = self.parameter("I", 10.0)
        self.noise_std = self.parameter("noise_std", 1.0)
        neurons.I = neurons.vector(mode=self.I)

    def forward(self, neurons):
        base = neurons.vector(mode=self.I)
        neurons.I = base + self.noise_std * torch.randn_like(base)
          
          
class PiecewiseCurrent(Behavior):

    def initialize(self, neurons):
        schedule = self.parameter("schedule", [(0, 0.0)])
        self.schedule = sorted(schedule, key=lambda s: s[0])
        neurons.I = neurons.vector(mode=float(self.schedule[0][1]))

    def forward(self, neurons):
        t = neurons.network.iteration * neurons.network.dt
        value = self.schedule[0][1]
        for t_start, val in self.schedule:
            if t >= t_start:
                value = val
        neurons.I = neurons.vector(mode=float(value))
