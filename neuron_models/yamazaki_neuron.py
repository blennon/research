from pylab import *

class YamazakiNeuron(object):
    
    def __init__(self, Vth, Cm, El, Eex, Einh, Eahp, gl, g_ex_, g_inh_,
                 g_ahp_, r_ex, r_inh, tau_ex, tau_inh, tau_ahp, I_spont,
                 dt):
        
        # scalar values
        self.Vth, self.Cm, self.El, self.Eex = Vth, Cm, El, Eex
        self.Einh, self.Eahp, self.gl, self.g_ex_ = Einh, Eahp, gl, g_ex_
        self.g_inh_, self.g_ahp_ = g_inh_, g_ahp_
        self.I_spont = I_spont
        
        # vector values
        self.r_ex = r_ex
        self.r_inh = r_inh
        self.tau_ex = tau_ex
        self.tau_inh = tau_inh
        self.tau_ahp = tau_ahp
        
        
        # decay
        # Yamazaki and Nagao implement the decay as exp(-dt/tau)
        # this is a numerical approximation to how BRIAN performs
        # decays with differential equations.  Using (1-dt/tau)
        # exactly matches the results from BRIAN (since this is a
        # Taylor series approximation to exp(-dt/tau).  If this isn't
        # used then the time constants have to be rescaled to get
        # Yamazaki and BRIAN to match up.
        if tau_ex is not None:
            self.decay_ex = exp(-dt/tau_ex)#1-dt/tau_ex #
        else:
            self.decay_ex = 0.
        if tau_inh is not None:
            self.decay_inh = exp(-dt/tau_inh)#1-dt/tau_inh #
        else:
            self.decay_inh = 0.
        self.decay_ahp = exp(-dt/tau_ahp)#1-dt/tau_ahp #
        
        # psp
        self.psp_ex = zeros_like(r_ex)
        self.psp_inh = zeros_like(r_inh)
        self.g_ex = 0.
        self.g_inh = 0.
        self.g_ahp = 0.
        
        self.u = El
        self.dt = dt
        self.just_spiked = False
        
        if self.I_spont > 0.:
            self.update_u()
        
    def update_u(self):
        dudt = (1./self.Cm)*(-self.gl     *              (self.u-self.El)
                             -self.g_ex_  * self.g_ex  * (self.u-self.Eex)
                             -self.g_inh_ * self.g_inh * (self.u-self.Einh)
                             -self.g_ahp_ * self.g_ahp * (self.u-self.Eahp)
                             +self.I_spont
                             )
        self.u += dudt*self.dt
    
    def update_psp(self, ex_spike, inh_spike, w_ex, w_inh):
        self.psp_ex = self.psp_ex * self.decay_ex + ex_spike
        self.psp_inh = self.psp_inh * self.decay_inh + inh_spike
        self.g_ex = w_ex * dot(self.psp_ex.T, self.r_ex)
        self.g_inh = w_inh * dot(self.psp_inh.T, self.r_inh)

    
    def update(self, ex_spike, inh_spike, w_ex, w_inh, reset_V=True):
        self.update_psp(ex_spike, inh_spike, w_ex, w_inh)
        
        if self.just_spiked:
            self.g_ahp = 1.0
            self.just_spiked = False
        else:
            self.g_ahp *= self.decay_ahp
        
        self.update_u()
        
        if self.u >= self.Vth:
            if reset_V:
                self.u = self.El
            self.just_spiked = True
            return 1.
        else:
            return 0.
