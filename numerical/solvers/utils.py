import numpy as np

def exact_solution(coord, npoin, time, icase):
    """
    Computes exact solution for test cases.
    
    Args:
        coord (array): Grid coordinates
        npoin (int): Number of points
        time (float): Current time
        icase (int): Test case number (1-6)
        
    Returns:
        tuple: (qe, u) Solution values and wave speed
    """
    # constants
    w = 1
    visc = 0
    h = 0
    xc = 0
    xmin = -1
    xmax = 1
    x1 = xmax-xmin
    sigma0 = 0.125
    rc = 0.125
    sigma = np.sqrt(sigma0**2 + 2*visc*time)
    u = w*x1
    
    # initialize
    qe = np.zeros(npoin)
    # print("Initial qe:", qe)  # Debug print
    
    timec = time - np.floor(time)
    
    for i in range(npoin):
        x = coord[i]
        xbar = xc + u*timec
        if(xbar > xmax):
            xbar = xmin + (xbar-xmax)
        r = x-xbar
        
        if(icase == 1):
            qe[i] = np.exp(-64.0*(x-xbar)**2)
        elif(icase == 2):
            if(abs(r) <= rc):
                qe[i] = 1
        elif(icase == 3):
            qe[i] = sigma0/sigma*np.exp(-(x-xbar)**2/(2*sigma**2))
        elif(icase == 4):
            if(abs(r) <= rc):
                qe[i] = 1
        elif(icase == 5):
            if(x <= xc):
                qe[i] = 1
        elif(icase == 6):
            qe[i] = np.sin(((x + 1)*np.pi)/2.0)
            # print(f"Just assigned qe[{i}] = {qe[i]}")  # Debug print
        
        # print(f"After iteration {i}, qe = {qe}")  # Debug print
    
    # print("Final qe before return:", qe)  # Debug print
    return qe, u

def L2_err_norm(nop, nelem, q0, qe):
    """
    Computes L2 error norm between numerical and exact solutions.
    
    Args:
        nop (int): Polynomial order
        nelem (int): Number of elements
        q0 (array): Numerical solution
        qe (array): Exact solution
        
    Returns:
        float: L2 error norm
    """
    Np = nelem*nop + 1
    num = 0
    den = 0
    for i in range(Np):
        num = num + (qe[i]-q0[i])**2
        den = den + qe[i]**2
    err = np.sqrt(num/den)

    return err

def L2normerr(q0, qe):
    num = np.norm(q0, qe)
    den = np.norm(qe)
    err = num/den
    return err


def compute_total_mass(q, Me, intma):
    """
    Compute total mass (integral of solution) using mass matrix
    """
    print(f'intma:\n {intma[0][:]}')
    mass = 0
    for e in range(len(intma[0][:])):  # loop over elements
        # Get local solution and mass matrix
        print(f'intma[e]: {intma[:,e]}')
        q_local = q[intma[:,e]]
        Me_local = Me[e]
        # Add contribution from this element
        mass += np.dot(np.dot(q_local, Me_local), q_local)
    return mass