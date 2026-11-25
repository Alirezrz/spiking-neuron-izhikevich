from pymonntorch import *

"""
a :time scale of u (small = slow recovery)
# b :sensitivity of u to subthreshold v (large = subthreshold oscillations, low-threshold spiking)
# c :after-spike reset value of v
# d :after-spike jump of u (large = strong adaptation)


v :membrane potential (mV), fast variable
u: membrane recovery variable, slow variable
I : input current


v[t] = v[t-1] + dt * (0.04*v[t-1]^2 + 5*v[t-1] + 140 - u[t-1] + I[t])
u[t] = u[t-1] + dt * a * (b*v[t] - u[t-1]


spikes:
if v >= threshold:
     v = c
     u = u + d

"""



class Izhikevich(Behavior):
    
    def initialize(self, neurons):
        
        # in this model we have 4 parameters -> a,b,c,d
         
        #a time scale of the recovery variable u :
        neurons.a=self.parameter("a",0.02)
        
        # b describes the sensitivity of the recovery variable u to the subthreshold fluctuations of the membrane potential v :
        neurons.b=self.parameter("b",0.2)
        
        # c describes the after-spike reset value of the membrane potential v
        neurons.c=self.parameter("c", -65.0)
        
        # d describes after-spike reset of the recovery variable u
        neurons.d=self.parameter("d",2)
        
        neurons.threshold = self.parameter("threshold", 30.0)
        v_init = self.parameter("v_init", -65.0)
        
        neurons.v=neurons.vector(mode=v_init)
        neurons.u= neurons.b*neurons.v
        
        # for each neuron if v > threshold then it spike
        neurons.spikes = neurons.v >= neurons.threshold
        
    def forward(self,neurons):
        dt = neurons.network.dt
        
        # check for spikes and handel those who have spike :
        neurons.spikes = neurons.v >= neurons.threshold
        neurons.v[neurons.spikes] = neurons.c
        neurons.u[neurons.spikes] += neurons.d
        
        #update v , then u:
        dv= 0.04 *neurons.v**2+ 5 * neurons.v + 140-neurons.u +neurons.I
        neurons.v += dt*dv
        
        neurons.u += dt*(neurons.a*(neurons.b* neurons.v - neurons.u))
        

        