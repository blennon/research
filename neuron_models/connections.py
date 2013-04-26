from pylab import *
from util import cartesian

def gr_to_pkj_connections(N_GO,N_GR,N_PKJ,go_span=4):
    '''
    Implementation of connection matrix from granule cells to purkinje cells

    Inspired by T. Yamazaki's code from Yamazaki and Nagao 2012
    from: https://github.com/blennon/Cerebellum/blob/master/okr.c

    N_GO: number of Golgi cells
    N_GR: number of Granule cells
    N_PKJ: number of Purkinje cells
    go_span: number of rows of GO cells to span when connecting from
             GR clusters to PKJs (each GR cluster of N_GR/N_GO cells
             corresponds to one GO).
    p   : probability of connecting a granule cell cluster to a Golgi cell
    
    This assumes a square grid of GO cells.  Each GO has a corresponding
    cluster of GR cells.  Each GR cells are indexed by what cluster they
    belong to, e.g. n:n+N_GR/N_GO for the nth GO.
    
    returns two arrays: pre and post synaptic indices
    '''
    connections = []
    w_GO = int(N_GO**.5)
    
    # arrange GRs into a grid of w_GO rows
    GR_grid = arange(N_GR).reshape(w_GO,w_GO*N_GR/N_GO)
    
    for pkj_ind in range(N_PKJ):
        n = int(ceil(float(w_GO)/N_PKJ)) # spacing between successive centers of row spans
        rows = arange(pkj_ind*n-go_span,pkj_ind*n+go_span+1) % w_GO # row spans
        for gr_ind in hstack(GR_grid[rows,:]):
            connections.append((gr_ind,pkj_ind))
    return map(array,zip(*set(connections)))

def gr_to_go_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 3, p=.05, wrap=False):
    '''
    Implementation of connection matrix from granule cells to golgi cells

    Inspired by T. Yamazaki's code from Yamazaki and Nagao 2012
    from: https://github.com/blennon/Cerebellum/blob/master/okr.c

    N_go: number of Golgi cells
    N_gr: number of Granule cells
    dist: distance in terms of number of Golgi cells (arranged in a grid)
          to consider connecting a cluster of 'n' granule cells to a Golgi
          cell, subject to probability 'p'
    p   : probability of connecting a granule cell cluster to a Golgi cell
    
    Note: This implementation assumes a square grid of neurons.
    
    returns two arrays: pre and post synaptic indices
    '''
    connections = []
    w = int(N_go**.5)
    n = N_gr / N_go
    go_grid = arange(w**2).reshape((w,w))
    
    # iterate over each golgi cell
    for i in range(w):
        for j in range(w):
            
            # get the indices of the surrounding (dist) golgi cells
            if wrap:
                arr_inds = cartesian((arange(i-dist,i+dist+1)%w,arange(j-dist,j+dist+1)%w))
                go_inds = go_grid[arr_inds[:,0],arr_inds[:,1]].reshape((2*dist+1,2*dist+1))
                 
            else:
                go_inds = go_grid[max(i-dist,0):min(i+dist+1,w),max(j-dist,0):min(j+dist+1,w)]
                
            # connect sets of corresponding granule cells to the center
            # golgi cell, selected at random from the set of surrounding golgi cells
            for go_ind in go_inds[rand(go_inds.shape[0],go_inds.shape[1]) < p]:
                for gr_ind in xrange(go_ind*n,go_ind*n + n):
                    connections.append((gr_ind,go_grid[i,j]))
    return map(array,zip(*set(connections)))

def go_to_gr_connections(N_go = 32**2, N_gr = 32**2 * 10**2, dist = 4, p=.025, wrap=False):
    '''
    Implementation of connection matrix from golgi cells to granule cells

    Inspired by T. Yamazaki's code from Yamazaki and Nagao 2012
    from: https://github.com/blennon/Cerebellum/blob/master/okr.c

    N_go: number of Golgi cells
    N_gr: number of Granule cells
    dist: distance in terms of number of Golgi cells (arranged in a grid)
          to consider connecting a Golgi cell to a cluster of 'n' granule 
          cells, subject to probability 'p'
    p   : probability of connecting a Golgi cell to a granule cell cluster
          within a window (dist x dist)
    
    Note: This implementation assumes a square grid of neurons.
    
    returns two arrays: pre and post synaptic indices
    '''
    connections = set()
    w = int(N_go**.5)
    n = N_gr / N_go
    go_grid = arange(N_go).reshape((w,w))
    
    # iterate over every golgi cell in a grid of golgi cells, index by (i,j) coordinates
    for i in range(w):
        for j in range(w):
            
            # get the single numeral indices of the surrounding golgi cells
            if wrap:
                go_arr_inds = cartesian((arange(i-dist,i+dist+1)%w,arange(j-dist,j+dist+1)%w))
                go_inds = go_grid[go_arr_inds[:,0],go_arr_inds[:,1]].reshape((2*dist+1,2*dist+1))
            else:
                go_inds = go_grid[max(i-dist,0):min(i+dist+1,w),max(j-dist,0):min(j+dist+1,w)]
                
            # randomly choose a subset of these surrounding golgi cells to connect
            for src_go_ind in go_inds[rand(go_inds.shape[0],go_inds.shape[1]) <= p]:
                
                # connect to glomeruli, i.e. the surrounding four granule cell clusters
                # surrounding is down and to the right
                if wrap:
                    go_gl_arr_inds = cartesian((arange(i,i+2)%w,arange(j,j+2)%w))
                    go_gl_inds = go_grid[go_gl_arr_inds[:,0],go_gl_arr_inds[:,1]]
                else:
                    go_gl_inds = go_grid[i:min(i+2,w),j:min(j+2,w)].flatten()
                    
                for go_gl_ind in go_gl_inds:
                    
                    # connect to all granule cells in a chosen cluster
                    for gr_ind in xrange(go_gl_ind*n, go_gl_ind*n + n):
                        connections.add((src_go_ind,gr_ind))
    return map(array,zip(*(connections)))