from pymonntorch import *




# ConstantCurrent : I = I_value
# StepCurrent     : I = 0 before t_start, I = I_value after
# NoisyCurrent    : I = I_value + noise_std * N(0, 1)



class ConstantCurrent( Behavior):
    def initialize(self, neurons):
        self.I=self.parameters("I",10)
        neurons.I=neurons.vector(mode=self.I)
    
    def forward(self,neurons):
        neurons.I=neurons.vector(mode=self.I)
            
            
            
            
class StepCurrent(Behavior):
    
    def initialize(self, neurons):
        self.I=self.parameters("I",10.0)
        
        self.start=self.parameters("t_start",0.0)
        
        neurons.I=neurons.vector(mode=0.0)
        
        
        
    def forward(self,neurons):
        t = neurons.network.iterations *neurons.network.dt
        neurons.I=neurons.vector(mode=self.I if t >= self.start else 0.0)    
        
        
class NoisyCurrent( Behavior):
    def initialize(self, neurons):
        self.I=self.parameters("I",10)
        self.noise_std=self.parameters("noise_std",1.0)
        neurons.I=neurons.vector(mode=self.I)
    
    def forward(self, neurons):
        noise = self.noise_std * torch.randn(neurons.size)
        neurons.I = neurons.vector(mode=self.I_value)+noise
            
          