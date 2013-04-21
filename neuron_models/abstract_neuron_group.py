from brian import *

class AbstractNeuronGroup(NeuronGroup):
    '''
    Extends NeuronGroup to include additional methods
    '''

    def get_parameters(self):
        '''Return a dictionary of parameter names and values'''
        raise NotImplementedError('Not implemented yet')
    
    def save_parameters(self, out_f):
        '''Save parameters to out_f file object with write permission'''
        for p,v in self.get_parameters().iteritems():
            out_f.write('%s\t%s\n' % (p,str(v)))        