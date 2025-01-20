import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches

# Functions

#Legendre-Poly
def leg_poly(p,x):

    L1, L1_1, L1_2 = 0, 0, 0
    L0, L0_1, L0_2 = 1, 0, 0

    for i in range(1,p+1):
        L2, L2_1, L2_2 = L1, L1_1, L1_2
        L1, L1_1, L1_2 = L0, L0_1, L0_2
        a  = (2*i-1)/i
        b  = (i-1)/i
        L0 = a*x*L1 - b*L2;
        L0_1 = a*(L1+x*L1_1)-b*L2_1
        L0_2 = a*(2*L1_1+x*L1_2) - b*L2_2

    return L0, L0_1, L0_2

### Routine for generating Legendre-Gauss_Lobatto points
def lgl_gen(P):
    # P is number of interpolation nodes. (P = order + 1)
    p = P-1 #Poly order
    ph = int(np.floor( (p+1)/2.0 ))

    lgl_nodes   = np.zeros(P)
    lgl_weights = np.zeros(P)

    for i in range(1,ph+1):
        x = np.cos((2*i-1)*np.pi/(2*p+1))

        for k in range (1,21):
            L0,L0_1,L0_2 = leg_poly(p,x)

            dx = -((1-x**2)*L0_1)/(-2*x*L0_1 + (1-x**2)*L0_2)
            x = x+dx

            if(abs(dx)<1.0e-20):
                break

        lgl_nodes[p+1-i]=x
        lgl_weights[p+1-i]=2/(p*(p+1)*L0**2)

    #Check for Zero root
    if(p+1 != 2*ph):
        x = 0
        L0, dum, dumm = leg_poly(p,x)
        lgl_nodes[ph] = x
        lgl_weights[ph] = 2/(p*(p+1)*L0**2)

    #Find remainder of roots via symmetry
    for i in range(1,ph+1):
        lgl_nodes[i-1] = -lgl_nodes[p+1-i]
        lgl_weights[i-1] =  lgl_weights[p+1-i]

    return lgl_nodes,lgl_weights

#Lagrange basis
def Lagrange_basis(P, Q, xlgl, xs):

    psi  = np.zeros([P,Q]) # PxQ matrix
    dpsi = np.zeros([P,Q])

    for l in range(Q):
        xl = xs[l]

        for i in range(P):
            xi = xlgl[i]
            psi[i][l]=1
            dpsi[i][l]=0

            for j in range(P):
                xj = xlgl[j]
                if(i != j):
                    psi[i][l]=psi[i][l]*((xl-xj)/(xi-xj))
                ddpsi=1
                if(i!=j):
                    for k in range(P):
                        xk=xlgl[k]
                        if(k!=i and k!=j):
                            ddpsi=ddpsi*((xl-xk)/(xi-xk))

                    dpsi[i][l]=dpsi[i][l]+(ddpsi/(xi-xj))

    return psi, dpsi

def create_diff_matrix(ngl,nq,wnq,psi,dpsi):
    # ngl is the number of LGL points
    # nq is the number of quadrature points
    # wnq are the lgl weights returned by lgl_gen(P) (returns xlgl, wlgl)
    # psi are the Lagrange basis functions returned by Lagrange_basis(P, Q, xlgl, xs) (returns psi, dpsi)
    # dpsi are the derivatives of psi returned by Lagrange_basis


    #initialize element derivative matrix
    e_diff = np.zeros([ngl,ngl])  #create an ngl x ngl size matrix


    #Do LGL integration
    for k in range(nq):
        wk = wnq[k]
        for i in range(ngl): #index over rows
            dhdx_i = dpsi[i][k]
            for j in range(ngl): #index over columns
                h_j = psi[j][k]
                e_diff[i][j] = e_diff[i][j] + wk*dhdx_i*h_j
#                 if(k==0 and i == 0 and j == 0):
#                     print(f'w{k} = {wk}')
#                     print(f'dpsi{i}{k} = {dpsi[i][k]}')
#                     print(f'psi{i}{k} = {psi[i][k]}')
#                     print(f'product{k} = {wk*dhdx_i*h_j}')
#                 print(f'i={i}, j={j}, D{i}{j}={e_diff[i][j]}')

    return e_diff

def create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi):
    # intma
    # coord
    # nelem is the number of elements
    # ngl is the number of LGL points
    # nq  is the number of quadrature points
    # wnq are the lgl weights returned by lgl_gen(P) (returns xlgl, wlgl)

    #initialize element mass matrix
    e_mass = np.zeros((nelem,ngl,ngl))
    x = np.zeros(ngl)

    for e in range(nelem):

        #store coordinates
        for i in range(ngl):
            I = int(intma[i][e])
            x[i] = coord[I]

        dx = x[-1]-x[0] #check this --> sould use len(x)?
        jac = dx/2.0

        #unstructured
        dx = x[-1]-x[0] #check this --> sould use len(x)?
        jac = dx/2.0

        #Do LGL integration
        for k in range(nq):
            wk=wnq[k]*jac

            for i in range(ngl):
                h_i=psi[i][k]
                for j in range(ngl):
                    h_j=psi[j][k]
                    e_mass[e][i][j] = e_mass[e][i][j] + wk*h_i*h_j

#             if (e == 0 and k==nq-1):
#                 print(f'e:{e}\n k:{k}\n i:{i}\n j:{j}\n')
#                 print(e_mass[e][i][j])
    return e_mass


def create_mass_matrix_vectorized(intma, coord, nelem, ngl, nq, wnq, psi):
    # Preallocate element mass matrix
    e_mass = np.zeros((nelem, ngl, ngl))

    for e in range(nelem):
        # Extract coordinates for this element
        x = coord[intma[:, e]]

        # Calculate Jacobian (simplified)
        dx = x[-1] - x[0]
        jac = dx / 2.0

        # Vectorized integration
        for k in range(nq):
            wk = wnq[k] * jac
            h_i = psi[:, k]
            h_j = psi[:, k]

            # Outer product for mass matrix computation
            e_mass[e] += wk * np.outer(h_i, h_j)

    return e_mass

def create_grid(ngl, nelem,npoin_cg, npoin_dg,xgl):
    # ngl is the number of LGL points
    # nelem is the number of elements
    # npoin_cg is .... n pointer for cg
    # npoin_dg is .... n pointer for dg
    # xgl are the lgl nodes

    #Initialize
    npin_cg = int(npoin_cg)
    npin_dg = int(npoin_dg)
    intma_dg=np.zeros([ngl,nelem], dtype = int)
    intma_cg=np.zeros([ngl,nelem], dtype = int)
    periodicity_cg = np.zeros(npin_cg, dtype = int)
    periodicity_dg = np.zeros(npin_dg, dtype = int)


    #Constants
    xmin = -1
    xmax = 1
    dx = (xmax-xmin)/nelem
    coord_cg = np.zeros(npoin_cg)
    coord_dg = np.zeros(npoin_dg)


    #generate COORD and INTMA for CG
    ip=0
    coord_cg[0]=xmin
    for e in range(nelem): #0,1,2,3
        x0 = xmin + (e)*dx
#         intma_cg[0][e] = ip+1 #produces same as MATLAB
        intma_cg[0][e] = ip
        for i in range(1, ngl): #1,2,3
#             print(f'e = {e}, i={i}')
            ip+=1
#             print(f'ip={ip}')
            coord_cg[ip]=(xgl[i]+1)*dx/2 + x0
#             print(f'dx/2')
#             print(f'coord: {coord_cg[ip]}')

#             intma_cg[i][e]=ip+1 #produces same as MATLAB
            intma_cg[i][e]=ip
#     print(f'cg coords:\n {coord_cg}')
#     print(f'intma_cg:\n{intma_cg}')

    #Generate periodicity pointer for CG
    for i in range(npoin_cg):
        periodicity_cg[i]=i
    periodicity_cg[-1] = periodicity_cg[0] # maybe use -1




    #generate COORD and INTMA for DG
    ip=0
    for e in range(nelem):
        for i in range(ngl):
#             ip+=1
            intma_dg[i][e] = ip
            ip+=1
#     print(f'intma_dg:\n {intma_dg}')
    for e in range(nelem):
        for i in range(ngl):
#             print(f'e = {e}, i={i}')
            ip_cg = intma_cg[i][e]
            ip_dg = intma_dg[i][e]
            coord_dg[int(ip_dg)] = coord_cg[int(ip_cg)];

    for i in range(npoin_dg):
        periodicity_dg[i]=i


    return  coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg


def create_grid_us(ngl, nelem, npoin_cg, npoin_dg, xgl, xelem):
    # ngl is the number of LGL points
    # nelem is the number of elements
    # npoin_cg is .... n pointer for cg
    # npoin_dg is .... n pointer for dg
    # xgl are the lgl nodes

    #xelem will be a level 0 array.  [-1, -0.4, 0, 0.4, 1]
    #break exelem down into elements in order to compute the jacobian for each elemnt.


    #Initialize
    npin_cg = int(npoin_cg)
    npin_dg = int(npoin_dg)
    intma_dg=np.zeros([ngl,nelem], dtype = int)
    intma_cg=np.zeros([ngl,nelem], dtype = int)
    periodicity_cg = np.zeros(npin_cg, dtype = int)
    periodicity_dg = np.zeros(npin_dg, dtype = int)


    #Constants
    xmin = -1
    xmax = 1
#     dx = (xmax-xmin)/nelem
    coord_cg = np.zeros(npoin_cg)
    coord_dg = np.zeros(npoin_dg)


    #generate COORD and INTMA for CG
    ip=0
    dx = 0
    x0 = xmin
    coord_cg[0]=xmin
    for e in range(nelem): #0,1,2,3
        x0 = x0 + dx
        dx = xelem[e+1]-xelem[e]
#         print(f'element: {e}, dx: {dx}')

#         print(f'x0: {x0}')
#         intma_cg[0][e] = ip+1 #produces same as MATLAB
        intma_cg[0][e] = ip
        for i in range(1, ngl): #1,2,3
#             print(f'e = {e}, i={i}')
            ip+=1
            # print(f'ip={ip}, i = {i}')

            coord_cg[ip]=(xgl[i]+1)*(dx/2) + x0

#             print(coord_cg[ip])

#             intma_cg[i][e]=ip+1 #produces same as MATLAB
            intma_cg[i][e]=ip
#     print(f'cg coords:\n {coord_cg}')
#     print(f'intma_cg:\n{intma_cg}')

    #Generate periodicity pointer for CG
    for i in range(npoin_cg):
        periodicity_cg[i]=i
    periodicity_cg[-1] = periodicity_cg[0] # maybe use -1




    #generate COORD and INTMA for DG
    ip=0
    for e in range(nelem):
        for i in range(ngl):
#             ip+=1
            intma_dg[i][e] = ip
            ip+=1
#     print(f'intma_dg:\n {intma_dg}')
    for e in range(nelem):
        for i in range(ngl):
#             print(f'e = {e}, i={i}')
            ip_cg = intma_cg[i][e]
            ip_dg = intma_dg[i][e]
            coord_dg[int(ip_dg)] = coord_cg[int(ip_cg)];

    for i in range(npoin_dg):
        periodicity_dg[i]=i


#     return  coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg
    return  coord_dg, intma_dg, periodicity_dg

def Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u):
    Fmat = np.zeros([npoin,npoin], dtype=int)

    for e in range(nelem):
        #Visit the left-most DOF of each element
        i=0
        I=intma[i][e]
        #shift left
        Im = I-1
        if(Im < 1):
            Im = npoin-1 #periodicity
        Fmat[I][Im]=-1

        #Visit right-most DOF of each element
        i=ngl-1
        I=intma[i][e]
        #shift left
        Ip = I+1
        if(Ip > npoin):
            Ip=1 #periodicity
        Fmat[I][I]=1

    Fmat = Fmat*u
    return Fmat

def Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, npoin):
    #Form global matrices
    M=np.zeros([npoin,npoin])
    D=np.zeros([npoin,npoin])

    for e in range(nelem):
        for i in range(ngl):
            ip = periodicity[intma[i][e]]
            for j in range(ngl):
                jp=periodicity[intma[j][e]]
                M[ip][jp]=M[ip][jp]+Me[e][i][j]
                D[ip][jp]=D[ip][jp]+u*De[i][j]

    return M,D


# def exact_solution(coord, npoin, time, icase):

#     # constatnts
#     w = 1
#     visc = 0
#     h = 0
#     xc = 0
#     xmin = -1
#     xmax = 1
#     x1 = xmax-xmin
#     sigma0 = 0.125
#     rc = 0.125
#     sigma = np.sqrt(sigma0**2 + 2*visc*time)
#     u = w*x1

#     #initiallize
#     qe=np.zeros(npoin)

#     timec = time - np.floor(time)

#     for i in range(npoin):
#         x = coord[i]
#         xbar = xc + u*timec
#         if(xbar > xmax):
#             xbar = xmin + (xbar-xmax)

#         r = x-xbar
#         if(icase == 1):
#             qe[i]=np.exp(-64.0*(x-xbar)**2)
#         elif(icase == 2):
#             if(abs(r) <= rc):
#                 qe[i]=1
#         elif(icase == 3):
#             qe[i]=sigma0/sigma*np.exp(-(x-xbar)**2/(2*sigma**2))
#         elif(icase == 4):
#             if(abs(r) <= rc):
#                 qe[i]=1
#         elif(icase == 5):
#             if(x <= xc):
#                 qe[i] = 1
#         elif(icase == 6):
#             qe[i] = np.sin(((x + 1)*np.pi)/2.0)
#             print(f'x_{i} = {x}: q_{i} = {qe[i]}')



#     return qe, u

def exact_solution(coord, npoin, time, icase):
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
    print("Initial qe:", qe)  # Debug print
    
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

def wave_solve(nop, nelem, xelem, integration_points, integration_type, space_method_type, icase, Courant_max, flux_type,time_final, max_level):
    ngl = nop + 1

    npoin_cg = nop*nelem + 1
    npoin_dg = ngl*nelem

    #Compute Interpolation and Integration Points
    xgl,wgl = lgl_gen(ngl)

    if (integration_points ==1):
        integration_text = 'LGL'
        if (integration_type ==1):
            noq = nop
        elif (integration_type ==2):
            noq = nop + 1
        nq = noq + 1
        xnq,wnq = lgl_gen(nq)

    # elif (integration_points == 2):

    psi, dpsi = Lagrange_basis(ngl,nq, xgl, xnq)
    row_sum_psi = sum(psi)
    row_sum_dpsi = sum(dpsi)
#     print(f'row sum psi {row_sum_psi}')
#     print(f'row sum dpsi {row_sum_dpsi}')

    #Create Grid
#     coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg  = create_grid(ngl,nelem,npoin_cg,npoin_dg,xgl)
#     coord_cg, coord_dg, intma_cg,  intma_dg,  periodicity_cg, periodicity_dg  = create_grid_us(ngl,nelem,npoin_cg,npoin_dg,xgl, xelem)
    coord_dg, intma_dg,  periodicity_dg  = create_grid_us(ngl,nelem,npoin_cg,npoin_dg,xgl, xelem)

    #Form Global Matrix and Periodic BC Pointers
    if (space_method_type == 'dg'):
        npoin = npoin_dg
        coord = coord_dg
        intma = intma_dg
        periodicity = periodicity_dg
#         print(f'{space_method_type} : {integration_text}')

    #Compute Exact Solution:
    time = 0
    qe, u = exact_solution(coord, npoin, time, icase)

    #Compute Courant Number
    dx = coord[1]-coord[0]
    dt = Courant_max*dx/u
    Courant = u*dt/dx
#     print(f'Courant = {Courant}, dt = {dt}, time_final = {time_final}')

    #Create Local/Element Mass and Differentiation Matrices
    Me = create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
    De = create_diff_matrix(ngl, nq, wnq, psi, dpsi)

    #Form Global Matrices
    Mmatrix, Dmatrix = Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, npoin)

    #Apply BCs
    if(flux_type == 2):
        Fmatrix = Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u)

    Rmatrix = Dmatrix - Fmatrix

    #Left-Multiply by Inverse Mass Matrix
    # Dmatrix_hat=Mmatrix\Rmatrix
    Dmatrix_hat = np.linalg.solve(Mmatrix,Rmatrix)

    #Initialize State Vector:
    q1=qe
    q0=qe
    qp=qe

    #Time Integration
    # q0, time, plots = ti_LSRK(q0, Dmatrix_hat, periodicity, time, time_final, dt)
    q0, time, plots, exact, grids, xelems = ti_LSRK_amr(q0, Dmatrix_hat, periodicity, xgl, xelem, wnq, psi, dpsi,u, time, time_final, dt, 1, max_level)
#     q0, time, plots = ti_LSRK_amr(q0, Dmatrix_hat, periodicity_dg, xgl, xelem, wnq, psi, dpsi, u, time, time_final, dt)

    #Compute Exact Solution:
    # qe, u = exact_solution(coord, npoin, time, icase)


    #Compute Norm
    L2_norm = norm(q0-qe)/norm(qe)
    err = L2_err_norm(nop, nelem, q0, qe)

    return coord, q0, qe, L2_norm, err, exact, grids, xelems

def next_level(xelem):
    m = len(xelem)
    out = np.zeros((2*m-1),dtype=xelem.dtype)
    out[::2] = xelem
    midpoints = (xelem[:-1] + xelem[1:]) / 2
    out[1::2]=midpoints
    return out

def level_arrays(xelem, max_level):
    levels = []
    levels.append(xelem)
    for i in range(max_level):
        next_lev = next_level(levels[i])
        levels.append(next_lev)

    return levels

def stacker(level):
    m = int(len(level))
    # out = np.zeros((2*m-1),dtype=xelem.dtype)
    out = np.zeros((2*m-1))
    out[::2] = level
    out[1::2]= out[2::2]
    stacker = out[:-1].reshape(int(m-1),2)
    return stacker

def vstacker(levels):
    stacks=[]
    for level in levels:
        stacks.append(stacker(level))
    vstack = np.vstack(stacks)
    return vstack

def ti_LSRK_amr(q0, Dhat, periodicity, xgl, xelem, wnq, psi, dpsi,u, time, time_final, dt, icase, max_level):
#     qe, u = exact_solution(coord, npoin, time, icase)

    ngl = int(len(xgl))
    nop = ngl - 1
    nq  = int(len(wnq))
    nelem = int(len(q0)/ngl)
    npoin_cg = nop*nelem + 1
    npoin_dg = ngl*nelem
    Npoin = npoin_dg
    label_mat, info_mat, active = forest(xelem, max_level)
    coord,  intma, periodicity  = create_grid_us(ngl,nelem,npoin_cg, Npoin ,xgl, xelem)
    qe, u = exact_solution(coord, npoin_dg, time, icase)


#     print(f'timesteps: {time_final/dt}')
    frames = np.ceil(time_final/dt)
#     print(f'frames: {frames}')
    cols = int(len(q0))
    rows = int(frames)
    plots = np.zeros((rows, cols))
    exact = []
    grids = []
    xelems = []
    # exact = np.zeros((rows, cols))


    RKA = np.array([0,
       (-567301805773) / (1357537059087),
       (-2404267990393) / (2016746695238),
       (-3550918686646) / (2091501179385),
       (-1275806237668) / (842570457699 )])

    RKB = np.array([(1432997174477) / (9575080441755 ),
       (5161836677717) / (13612068292357),
       (1720146321549) / (2090206949498 ),
       (3134564353537) / (4481467310338 ),
       (2277821191437) / (14882151754819)])

    RKC = np.array([0,
       (1432997174477) / (9575080441755),
       (2526269341429) / (6820363962896),
       (2006345519317) / (3224310063776),
       (2802321613138) / (2924317926251)])

    Npoin = len(q0) #also use for npoin_dg
#     print(f'Npoin = {Npoin}')
    dq = np.zeros(Npoin)
    qp=q0
    stages=len(RKA)
#     print(f'stages = {stages}')

    #time integration:
    anim = 0
    grid = xelem
    while (time < time_final):
        time = time + dt
        if (time > time_final):
            time = time -dt
            dt = time_final-time
            time = time+dt

        #~~~~~~~~~~~~~~~~

        #Insert refinement routines here
        level = 0
        while(level < max_level):
          #call marking routine
          # print(f'\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
          # print(f'timestep: {anim}, level: {level}')
          # refs = mark(active, label_mat, info_mat, coord, intma, level, max_level, qe)
          # print(f'\n\ntimestep: {anim}, level: {level}')
          # print(f'active: {active}')


          print(f'\n\ntimestep: {anim}, level: {level}')
          print(f'now marking: active grid going into marking: {active}')
          refs,defs,marks = mark(active, label_mat, info_mat, coord, intma, level, max_level, qe)
          print(f'\ntimestep: {anim}\n active: {active}\n marks: {marks}\n refs: {refs}\n defs:{defs}')
          # print(f'refs: {refs}')
          # print(f'defs: {defs}')
          # if any(mark < 0 for mark in marks):
          #   print(f'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')


          # print(f'marks: {marks}')

          # Refine elements marked for refinement:
          # if(anim == 48):
          # print(f'pre refinement')
          # print(f'xelem: {xelem}')
          # print(f'active: {active}')
          # print(f'marks: {marks}')

          
          print(f'adapting mesh')
          grid, active, marks, nelem, npoin_cg, npoin_dg = adapt_mesh(nop, grid, active, label_mat, info_mat, marks)
          print(f'creating new grid')
          coord,  intma, periodicity  = create_grid_us(ngl, nelem, npoin_cg, npoin_dg,xgl, grid)

        #   refined, active, marks, new_nelem, new_npoin_cg, new_npoin_dg = refine(nop, xelem, active, label_mat, info_mat, refs, marks)
        #   xelem = refined

        #   ref_coord,  intma, periodicity  = create_grid_us(ngl,new_nelem,new_npoin_cg,new_npoin_dg,xgl, xelem)
        #   coord = ref_coord
        #   # if(anim == 48):
        #   # print(f'post refinement, pre coarsening:')
        #   # print(f'xelem: {xelem}')
        #   # print(f'active: {active}')
        #   # print(f'marks: {marks}')
        #     # print(f'coord: {coord}')



        #   # Coarsen elements marked for coarsening:
        #   derefined, active, marks, new_nelem, new_npoin_cg, new_npoin_dg = derefine(nop, xelem, active, label_mat, info_mat, defs, marks)
        #   xelem = derefined
        #   def_coord,  intma, periodicity  = create_grid_us(ngl,new_nelem,new_npoin_cg,new_npoin_dg,xgl, xelem)
        #   coord = def_coord

        #   # if(anim == 48):
        #   # print(f'post coarsening:')
        #   # print(f'xelem: {xelem}')
        #   # print(f'active: {active}')
        #   # print(f'marks: {marks}')
        #     # print(f'coord: {coord}')


          qe, u = exact_solution(coord, npoin_dg, time, icase)

          level +=1
          # print('\n')



#         Me = create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
#         De = create_diff_matrix(ngl, nq, wnq, psi, dpsi)
#         Mmatrix, Dmatrix = Matrix_DSS(Me, De, u, intma, periodicity, ngl, nelem, Npoin)
#         Fmatrix = Fmatrix_upwind_flux(intma, nelem, npoin, ngl, u)
#         Rmatrix = Dmatrix - Fmatrix
#         Dhat = np.linalg.solve(Mmatrix,Rmatrix)


        #~~~~~~~~~~~~~~~

        #RK stages
        for s in range(stages):
            #Create RHS Matrix
            R = Dhat@qp #only valid for cg
            #solve system
            for I in range(Npoin):
#                 print(f'I: {I}')
#                 print(f'dt: {dt}')
                dq[I] = RKA[s]*dq[I] + dt*R[I]
                qp[I] = qp[I] + RKB[s]*dq[I]

            if(periodicity[-1] == periodicity[0]):
                qp[-1]=qp[0]


        q0 = qp
        # qe, u = exact_solution(coord, npoin, time, icase)
#         print(f'timestep: {anim}\n plot: {qp}')
#         if (anim%10 == 0):
        plots[anim][:] = q0
        # exact[anim][:] = qe
        exact.append(qe)
        grids.append(coord)
        xelems.append(grid)
        # print(f'\ntimestep: {anim}\n active: {active}\n marks: {marks}')
#         print(f'plots[i]: {plots[anim]}')
        anim +=1





    return q0, time, plots, exact, grids, xelems



def forest(xelem0, max_level):
    # xelem0 is the initial, level 0 grid
    # max_level is the maximum refinement level

    # this routine will take in an inital grid and max level value and return a labeling array and
    # tree info array whos values are used for refinement


    #check data type of xelem0 --> must be float
    assert(xelem0.dtype == 'float64'), "grid data type must be float64"


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #Creat labeling martix
    levels=max_level+1
    elems0 = len(xelem0)-1
    # print(f'initial number of elements: {elems0}')
    rows = elems0
    lmt = 0
    elems = np.zeros(levels, dtype = int)
    elems[0]=len(xelem0)-1
    for i in range(levels-1):
        a = 2**(i+1)*elems0
        rows += a
    lmt = rows - a #elements ater this value have no children
    cols = 4
    label_mat = np.zeros([rows, cols], dtype = int)
    info_mat  = np.zeros([rows, 2])
    ctr = 2
    for j in range(rows):
        div = len(xelem0)-1
        label_mat[j][0], info_mat[j][0] = j + 1, j + 1
        if(j<div):
            label_mat[j][1], info_mat[j][1] = j//div, int(j//div)
        else:
            label_mat[j][1], info_mat[j][1] = (ctr)//2, int(ctr//2)
            ctr+=1
        if (j<lmt):
            label_mat[j][2] = div +(2*j+1)
            label_mat[j][3] = div + 2*(j+1)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #create tree-info matrix
    levels = level_arrays(xelem0, max_level)
    vstack = vstacker(levels)
    coord_mat = np.hstack((info_mat, vstack))
#     print(f'vstack:\n{vstack}')
#     print(f'shape: {np.shape(vstack)}')
    active_grid=np.zeros(len(xelem0)-1, dtype = int)
    for k in range(len(active_grid)):
        active_grid[k]=k+1

    return label_mat, coord_mat, active_grid

def elem_info(elem, label_mat):
    parent = label_mat[elem-1][1]
    c1 = label_mat[elem-1][2]
    c2 = label_mat[elem-1][3]
    print(f'\n\n element number {elem} has parent {parent} and children {c1} and {c2}')
    if (parent != 0):
      # find sibling
        if label_mat[elem-2][1] == parent:
            sib = elem-1
        elif label_mat[elem][1] == parent:
            sib = elem+1
        print(f'eleemnt {elem} has sibling {sib}')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~` this mark function has been optimized by Cursor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
def mark(active_grid, label_mat, info_mat, coord, intma, cur_level, max_level, q):
    """
    Optimized marking routine for mesh refinement/coarsening.
    
    Args:
        active_grid: Array of active element indices
        label_mat: Matrix containing parent-child relationships
        info_mat: Matrix containing element information
        coord: Grid coordinates
        intma: Element connectivity matrix
        cur_level: Current refinement level
        max_level: Maximum refinement level
        q: Solution values
        
    Returns:
        tuple: (refinement indices, coarsening indices, marking flags)
    """
    n_active = len(active_grid)
    marks = np.zeros(n_active, dtype=int)
    refs = []
    defs = []
    
    # Pre-compute label matrix lookups
    parents = label_mat[active_grid - 1, 1]
    children = label_mat[active_grid - 1, 2:4]
    
    # Process each active element
    for idx, (elem, parent) in enumerate(zip(active_grid, parents)):
        # Get element solution values
        elem_nodes = intma[:, idx]
        elem_sols = q[elem_nodes]
        max_sol = np.max(elem_sols)
        
        # Check refinement criteria
        if max_sol >= 0.5 and children[idx, 0] != 0:
            refs.append(elem)
            marks[idx] = 1
            continue
            
        # Check coarsening criteria
        if max_sol < 0.5 and parent != 0:
            # Find sibling
            sibling = None
            if elem > 1 and label_mat[elem-2, 1] == parent:
                sibling = elem - 1
                sib_idx = idx - 1
            elif elem < len(label_mat) and label_mat[elem, 1] == parent:
                sibling = elem + 1
                sib_idx = idx + 1
                
            # Verify sibling status
            if sibling in active_grid:
                sib_nodes = intma[:, sib_idx]
                sib_sols = q[sib_nodes]
                
                # Mark for coarsening if sibling also qualifies
                if np.max(sib_sols) < 0.5 and sibling not in defs:
                    marks[idx] = marks[sib_idx] = -1
                    defs.extend([elem, sibling])
    
    return refs, defs, marks

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~` the function below is mine before optimization suggestions via Cursor ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~`
# def mark(active_grid, label_mat, info_mat, coord, intma, cur_level, max_level, q):
#     marks = np.zeros(len(active_grid), dtype = int)
#     refs = []
#     defs = []

#     #index through active_grid and examine refinement criteria
#     ind = 0
#     for e in active_grid:
#         # get element family info:
#         # elem_info(e,label_mat)
#         parent = label_mat[e-1][1]
#         child_1 = label_mat[e-1][2]
#         child_2 = label_mat[e-1][3]
#         # print(f'\n element number {e} has parent {parent} and children {child_1} and {child_2}')

#         #access the element
#         # print(f'active grid: {active_grid}')
#         # print(f'evaluating element {e} for refinement:')
#         # print(f'the nodes in element {e} have the following global indices:')
#         index = np.where(active_grid == e)[0][0]
#         # print(f'index: {index}')
#         # print(f'ctr: {ind}')
#         # print(f'intma:')
#         # display(intma)
#         # print(f'index at e:{index}')
#         # print(f'intma at index: {intma[:,index]}')


#         elem_nodes = intma[:,index]
#         # elem_nodes = intma[:,e-1]
#         elem_sols = q[elem_nodes]
#         # print(f'element {e} has global solution values: {elem_sols}')
#         # print(f'element {e} has global x values: {coord[elem_nodes]}')

#         #check for refinement:
#         if max(elem_sols) >= 0.5 and child_1 != 0:
#           # print(f'element {e}, with index {index} is marked for refinement.')
#           refs.append(e)
#           marks[index] = 1

#         #check for derefinement: Only derefine if previously refined. Else, do nothing.
#         elif(max(elem_sols) < 0.5 and parent != 0):
#           # Identify sibling and check it's status.
#           # print(f'element {e} marked for potential derefinement')
#           if label_mat[e-2][1] == parent:
#             sib = e-1
#           elif label_mat[e][1] == parent:
#             sib = e+1
#           # print(f'element {e} has sibling {sib}')
#           #check whether sibling is active. If not action should be zero.

#           if sib in active_grid:
#              pass
#             #  print(f'sibling {sib} is active')
#           else:
#             # print(f'sibling {sib} is NOT active')
#             marks[index]=0
#             ind += 1
#             continue
#           if(e < sib):
#               sib_index = ind + 1
#           elif(e > sib):
#               sib_index = ind - 1




#           # sib_index = np.where(active_grid == sib)[0][0]
#           sib_nodes = intma[:,sib_index]
#           sib_sols = q[sib_nodes]

#           #Check whether sibling also meets derefinement criteria. If so, derefine both. If not, derefine neither.
#           # if(max(sib_sols) >= 0.5):
#           #   # print(f'sibling {sib} should not be derefined.')
#           #   # marks[index]=0
#           #   # marks[sib_index]=0
#           if(max(sib_sols) < 0.5):
#               # print(f'sibling {sib} should be derefined.')
#               if sib not in defs:
#                   marks[index]= -1
#                   marks[sib_index]=-1
#                   defs.append(e)
#                   defs.append(sib)
#               else:
#                 pass
#                 # print(f'element {e} already marked for derefinement during sibling\'s marking.')

#         ind+=1

#         # else:
#           # print(f'element {e} is NOT MARKED')




#     return refs, defs, marks
#     # print(f'marked')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~ Cursor optimized refine function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def refine(nop, cur_grid, active, label_mat, info_mat, refs, marks):
#     """
#     Optimized mesh refinement routine that processes elements marked for refinement.
    
#     Args:
#         nop: Number of points
#         cur_grid: Pre-refined grid coordinates
#         active: Active cells array
#         label_mat: Matrix containing parent-child relationships
#         info_mat: Matrix containing cell information
#         refs: Array of elements marked for refinement
#         marks: Array indicating which elements to refine
    
#     Returns:
#         tuple: (refined grid, active cells, marks, new element count, 
#                new CG point count, new DG point count)
#     """
#     ngl = nop + 1
    
#     # Early exit if no refinement needed
#     if not np.any(np.array(marks) > 0):
#         new_nelem = len(active)
#         return (cur_grid, active, marks, new_nelem, 
#                 nop * new_nelem + 1, ngl * new_nelem)
    
#     # Process refinements in batches
#     refinement_indices = np.where(np.array(marks) > 0)[0]
    
#     # Pre-compute child information for all refinements
#     for idx in refinement_indices:
#         elem = active[idx]
#         print(f'refining element {elem}')
#         parent_idx = elem - 1
#         c1, c2 = label_mat[parent_idx][2:4]
#         c1_r = info_mat[c1-1][3]
        
#         # Update grid
#         ref_index = idx + 1
#         cur_grid = np.insert(cur_grid, ref_index, c1_r)
        
#         # Update active cells and marks
#         print(f'active before refinement {active}')
#         active = np.concatenate([
#             active[:idx],
#             [c1, c2],
#             active[idx+1:]
#         ])
#         print(f'active after refinement {active}')

#         print(f'marks before refinement {marks}')
#         marks = np.concatenate([
#             marks[:idx],
#             [0, 0],
#             marks[idx+1:]
#         ])
#         print(f'marks after refinement {marks}')
#         refinement_indices = np.where(np.array(marks) > 0)[0]

    
#     # Calculate new dimensions
#     new_nelem = len(active)
#     new_npoin_cg = nop * new_nelem + 1
#     new_npoin_dg = ngl * new_nelem
    
#     return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def refine(nop, cur_grid, active, label_mat, info_mat, refs, marks):
    """
    Optimized mesh refinement routine that processes elements marked for refinement.
    Handles shifting indices when elements are refined.
    
    Args:
        nop: Number of points
        cur_grid: Pre-refined grid coordinates
        active: Active cells array
        label_mat: Matrix containing parent-child relationships
        info_mat: Matrix containing cell information
        refs: Array of elements marked for refinement
        marks: Array indicating which elements to refine
    
    Returns:
        tuple: (refined grid, active cells, marks, new element count, 
               new CG point count, new DG point count)
    """
    ngl = nop + 1
    
    # Early exit if no refinement needed
    if not np.any(np.array(marks) > 0):
        new_nelem = len(active)
        return (cur_grid, active, marks, new_nelem, 
                nop * new_nelem + 1, ngl * new_nelem)
    
    # Process refinements one at a time to handle shifting indices
    i = 0
    while i < len(marks):
        if marks[i] <= 0:
            i += 1
            continue
            
        # Get element and its children
        elem = active[i]
        parent_idx = elem - 1
        c1, c2 = label_mat[parent_idx][2:4]
        c1_r = info_mat[c1-1][3]
        
        # Update grid
        cur_grid = np.insert(cur_grid, i + 1, c1_r)
        
        # Update active cells and marks
        active = np.concatenate([
            active[:i],
            [c1, c2],
            active[i+1:]
        ])
        
        marks = np.concatenate([
            marks[:i],
            [0, 0],
            marks[i+1:]
        ])
        
        # Skip the newly added element in the next iteration
        i += 2
    
    # Calculate new dimensions
    new_nelem = len(active)
    new_npoin_cg = nop * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    
    return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg

# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ my original refine function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# def refine(nop, cur_grid, active, label_mat, info_mat, refs, marks):
#     # cur_grid is is the pre-refined grid of coordinates at time of function call
#     # active is the active cells at time of function call
#     # refs is an array of elements marked for refinement. This function will refine those elements.


#     # print(f'prerefined grid: {cur_grid}\n')
#     ngl = nop + 1
#     ctr = 0
#     ctr_max = len(refs)
#     # print(f'ctr_max = {len(refs)}')
#     # for elem in refs:
#     if any(mark > 0 for mark in marks):
#     # if any(marks):
#         # for elem,mark in zip(active, marks):
#         for i in range(len(active)):
#             mark = marks[ctr]
#             elem = active[ctr]
#             if mark == 0 or mark == -1:
#                 ctr +=1
#                 continue
#             #if mark != 0, then the following lines will be executed

#             # print(f'ctr: {ctr}')
#             # print(f'elem: {elem}')
#             # print(f'mark: {mark}')

#             # print(f'refine element: {elem}')
#             parent = label_mat[elem-1][1]
#             c1 = label_mat[elem-1][2]
#             c2 = label_mat[elem-1][3]
#             c1_l, c1_r = info_mat[c1-1][2],info_mat[c1-1][3]
#             c2_l, c2_r = info_mat[c2-1][2],info_mat[c2-1][3]

#             ref_index = np.where(active == elem)[0][0] + 1
#             # print(f'element {elem} is at {ref_index-1}. That is, index {ref_index-1} of active is {active[ref_index-1]}')
#             ref = np.insert(cur_grid, ref_index, c1_r)
#             cur_grid = ref
#             # Replace the value 3 with 30 and 31
#             index = np.where(active == elem)[0][0]
#             # print(f'here 4')
#             # print(f'index: {index}')
#             # print(f'active grid before refining element {elem} is {active}')
#             active = np.delete(active,index)
#             # print(f'here 5')
#             # print(f'marks before refining element {elem} is {marks}')
#             active = np.insert(active, index, [c1, c2])
#             # print(f'new active grid after refining element {elem} is : {active}')
#             marks = np.delete(marks,index)
#             marks = np.insert(marks, index, [0, 0])
#             # print(f'new marks after refining element {elem} is : {marks}')
#             new_nelem = len(active)
#             #     print(type(ref))
#             new_npoin_cg = nop*new_nelem + 1
#             new_npoin_dg = ngl*new_nelem
#             ctr+=2
#             # print(f'\n element number {elem} has parent {parent} and children {c1} and {c2}')

#     else:
#         # print(f'no elements to refine')
#         ref = cur_grid
#         new_nelem = len(active)
#         new_npoin_cg = nop*new_nelem + 1
#         new_npoin_dg = ngl*new_nelem
#     # if(ctr == ctr_max):
#     #   # print(f'no elements refined')
#     #   ref = cur_grid
#     #   new_nelem = len(active)
#     # #     print(type(ref))
#     #   new_npoin_cg = nop*new_nelem + 1
#     #   new_npoin_dg = ngl*new_nelem


#     return ref, active, marks, new_nelem, new_npoin_cg, new_npoin_dg
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~my original derefine function ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
# def derefine(nop, cur_grid, active, label_mat, info_mat, defs, marks):
#     # cur_grid is is the pre-refined grid of coordinates at time of function call
#     # active is the active cells at time of function call
#     # defs is an array of elements marked for refinement. This function will refine those elements.


#     # ~~~~~~~~~~~> defs should already be paired up as siblings
#     ngl = nop + 1
#     ctr = 0
#     ctr_max = len(marks)
#     # print(f'ctr_max = {len(refs)}')
#     # for elem in refs:
#     if any(mark < 0 for mark in marks):
#     # if any(marks):
#         # for elem,mark in zip(active, marks):
#         for i in range(ctr_max-1):
#             if all(m == 0 for m in marks):
#                     break
#             mark = marks[ctr]
#             elem = active[ctr]
#             if mark == 0 or mark == 1:
#                 # print(f'element {elem} is not marked for refinement')
#                 ctr +=1
#                 continue

#             #if mark != 0, then the following lines will be executed
#             else:
#                 # print(f'ctr: {ctr}')
#                 # print(f'elem: {elem}')
#                 # print(f'mark: {mark}')
#                 # print(f'derefine element: {elem}')
#                 parent = label_mat[elem-1][1]
#                 c1 = label_mat[elem-1][2]
#                 c2 = label_mat[elem-1][3]
#                 c1_l, c1_r = info_mat[c1-1][2],info_mat[c1-1][3]
#                 c2_l, c2_r = info_mat[c2-1][2],info_mat[c2-1][3]

#                 def_index = np.where(active == elem)[0][0] + 1
#                 # print(f'element {elem} is at {def_index-1}. That is, index {def_index-1} of active is {active[def_index-1]}')
#                 if label_mat[elem-2][1] == parent:
#                     sib = elem-1
#                 elif label_mat[elem][1] == parent:
#                     sib = elem+1
#                 # print(f'element {elem} has sibling {sib}')
#                 # print(f'element {elem} has sibling {active[ctr+1]}')

#                 # cur_grid.pop(def_index + 1)
#                 # print(f'dleting grid element: {cur_grid[def_index]}')
#                 deref = np.delete(cur_grid, def_index)
#                 cur_grid = deref
#                 # cur_grid = ref
#                 # Replace the value 3 with 30 and 31
#                 index = np.where(active == elem)[0][0]
#                 # print(f'here 4')
#                 # print(f'index: {index}')
#                 # print(f'active grid before derefining element {elem} and {sib} is {active}')
#                 active = np.delete(active,index)
#                 active = np.delete(active,index)
#                 # print(f'here 5')
#                 # print(f'marks before derefining element {elem} and {sib} is {marks}')
#                 active = np.insert(active, index, parent)
#                 # print(f'new active grid after derefining element {elem} and {sib} is : {active}')
#                 marks[index] = 0
#                 marks =  np.delete(marks,index+1)
#                 ctr_max -=1
#                 # print(f'new marks after derefining element {elem} and {sib} is : {marks}')
                
#                 new_nelem = len(active)
#                 #     print(type(ref))
#                 new_npoin_cg = nop*new_nelem + 1
#                 new_npoin_dg = ngl*new_nelem
#                 deref = cur_grid
#                 ctr+=1
#             # print(f'\n element number {elem} has parent {parent} and children {c1} and {c2}')

#     else:
#         # print(f'no elements to derefine')
#         deref = cur_grid
#         new_nelem = len(active)
#         new_npoin_cg = nop*new_nelem + 1
#         new_npoin_dg = ngl*new_nelem


#     return deref, active, marks, new_nelem, new_npoin_cg, new_npoin_dg
# #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Cursor optimized derefine~~~~~~~~~~~~~~~~~~~~~~``

def derefine(nop, cur_grid, active, label_mat, info_mat, defs, marks):
    """
    Optimized mesh coarsening routine that processes elements marked for derefinement.
    Handles shifting indices when elements are derefined.
    
    Args:
        nop: Number of points
        cur_grid: Current grid coordinates
        active: Active cells array
        label_mat: Matrix containing parent-child relationships
        info_mat: Matrix containing cell information
        defs: Array of elements marked for derefinement
        marks: Array indicating which elements to derefine
    
    Returns:
        tuple: (derefined grid, active cells, marks, new element count, 
               new CG point count, new DG point count)
    """
    ngl = nop + 1
    
    # Early exit if no derefinement needed
    if not np.any(np.array(marks) < 0):
        new_nelem = len(active)
        return (cur_grid, active, marks, new_nelem, 
                nop * new_nelem + 1, ngl * new_nelem)
    
    # Process derefinements one at a time
    i = 0
    while i < len(marks):
        if marks[i] >= 0:
            i += 1
            continue
            
        elem = active[i]
        parent = label_mat[elem-1][1]
        
        # Find sibling
        if label_mat[elem-2][1] == parent and i > 0 and marks[i-1] < 0:
            # Sibling is previous element
            sib_idx = i - 1
            min_idx = sib_idx
        elif i + 1 < len(marks) and label_mat[elem][1] == parent and marks[i+1] < 0:
            # Sibling is next element
            sib_idx = i + 1
            min_idx = i
        else:
            # No valid sibling found for derefinement
            i += 1
            continue
            
        # Remove grid point between elements
        cur_grid = np.delete(cur_grid, min_idx + 1)
        
        # Update active cells and marks
        active = np.concatenate([
            active[:min_idx],
            [parent],
            active[min_idx+2:]
        ])
        
        marks = np.concatenate([
            marks[:min_idx],
            [0],
            marks[min_idx+2:]
        ])
        
        # Continue checking from the position after the derefined pair
        i = min_idx + 1
    
    # Calculate new dimensions
    new_nelem = len(active)
    new_npoin_cg = nop * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    
    return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg



#~~~~~~~~~~~~~~~~~~~ single refine/coarsen routine ~~~~~~~~~~~~~~~~~~~~~~~~
def adapt_mesh(nop, cur_grid, active, label_mat, info_mat, marks):
    """
    Unified mesh adaptation routine that handles both refinement and derefinement.
    
    Args:
        nop: Number of points
        cur_grid: Current grid coordinates
        active: Active cells array
        label_mat: Matrix containing parent-child relationships
        info_mat: Matrix containing cell information
        marks: Array indicating refinement (-1: derefine, 0: no change, 1: refine)
    
    Returns:
        tuple: (adapted grid, active cells, marks, new element count, 
               new CG point count, new DG point count)
    """
    ngl = nop + 1
    
    # Early exit if no adaptation needed
    if not np.any(marks):
        new_nelem = len(active)
        return (cur_grid, active, marks, new_nelem, 
                nop * new_nelem + 1, ngl * new_nelem)
    
    # Process adaptations one at a time
    i = 0
    while i < len(marks):
        if marks[i] == 0:
            i += 1
            continue
            
        if marks[i] > 0:
            # Handle refinement
            elem = active[i]
            print(f'refining element {elem}')
            parent_idx = elem - 1
            c1, c2 = label_mat[parent_idx][2:4]
            print(f'elemenet {elem} has children {c1} and {c2} ')
            c1_r = info_mat[c1-1][3]
            
            # Update grid
            cur_grid = np.insert(cur_grid, i + 1, c1_r)
            
            # Update active cells and marks
            active = np.concatenate([
                active[:i],
                [c1, c2],
                active[i+1:]
            ])
            
            marks = np.concatenate([
                marks[:i],
                [0, 0],
                marks[i+1:]
            ])
            
            # Skip the newly added element
            i += 2
            
        else:  # marks[i] < 0
            # Handle derefinement
            elem = active[i]
            parent = label_mat[elem-1][1]
            
            # Find sibling
            if label_mat[elem-2][1] == parent and i > 0 and marks[i-1] < 0:
                # Sibling is previous element
                sib_idx = i - 1
                min_idx = sib_idx
            elif i + 1 < len(marks) and label_mat[elem][1] == parent and marks[i+1] < 0:
                # Sibling is next element
                sib_idx = i + 1
                min_idx = i
            else:
                # No valid sibling found for derefinement
                i += 1
                continue
                
            # Remove grid point between elements
            cur_grid = np.delete(cur_grid, min_idx + 1)
            
            # Update active cells and marks
            active = np.concatenate([
                active[:min_idx],
                [parent],
                active[min_idx+2:]
            ])
            
            marks = np.concatenate([
                marks[:min_idx],
                [0],
                marks[min_idx+2:]
            ])
            
            # Continue checking from the position after the derefined pair
            i = min_idx + 1
    
    # Calculate new dimensions
    new_nelem = len(active)
    new_npoin_cg = nop * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    
    return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg



# Projection Matrices

# def S_psi(P, Q, xlgl, xs, c):

#     # P = ngl --> number of lgl nodes
#     # Q = nq  --> number of quadrature points

#     psi = np.zeros([P,Q]) # PxQ matrix
#     dpsi = np.zeros([P,Q])


#     for l in range(Q):
#         xl = xs[l]
#         # z1 = 0.5*xl - 0.5
#         # z2 = 0.5*xl + 0.5
#         if c == 1:
#             zl = 0.5*xl - 0.5
#         elif c == 2:
#             zl = 0.5*xl + 0.5

#         for i in range(P):
#             xi = xlgl[i]
#             psi[i][l]=1
#             dpsi[i][l]=0

#             for j in range(P):
#                 xj = xlgl[j]
#                 if(i != j):
#                     psi[i][l]=psi[i][l]*((zl-xj)/(xi-xj))
#                 ddpsi=1
#                 if(i!=j):
#                     for k in range(P):
#                         xk=xlgl[k]
#                         if(k!=i and k!=j):
#                             ddpsi=ddpsi*((zl-xk)/(xi-xk))

#                     dpsi[i][l]=dpsi[i][l]+(ddpsi/(xi-xj))

#     return psi, dpsi

def S_psi(P, Q, xlgl, xs, c):

    # P = ngl --> number of lgl nodes
    # Q = nq  --> number of quadrature points
    # xlgl are LGL nodes
    # xs are the quadrature points.

    psi = np.zeros([P,Q]) # PxQ matrix
    dpsi = np.zeros([P,Q])


    for l in range(Q):
        xl = xs[l]
        # z1 = 0.5*xl - 0.5
        # z2 = 0.5*xl + 0.5
        if c == 1:
            zl = 0.5*xl - 0.5
        elif c == 2:
            zl = 0.5*xl + 0.5

        for i in range(P):
            xi = xlgl[i]
            zi = xi
            # if c == 1:
            #     zi = 0.5*xi - 0.5
            # elif c == 2:
            #     zi = 0.5*xi + 0.5
            psi[i][l]=1
            dpsi[i][l]=0

            for j in range(P):
                xj = xlgl[j]
                zj = xj
                # if c == 1:
                #     zj = 0.5*xj - 0.5
                # elif c == 2:
                #     zj = 0.5*xj + 0.5
                if(i != j):
                    psi[i][l]=psi[i][l]*((zl-zj)/(zi-zj))
                ddpsi=1
                if(i!=j):
                    for k in range(P):
                        xk=xlgl[k]
                        zk = xk
                        # if c == 1:
                        #     zk = 0.5*xk - 0.5
                        # elif c == 2:
                        #     zk = 0.5*xk + 0.5
                        if(k!=i and k!=j):
                            ddpsi=ddpsi*((zl-zk)/(zi-zk))

                    dpsi[i][l]=dpsi[i][l]+(ddpsi/(zi-zj))

    return psi, dpsi



def create_S_matrix(intma, coord, nelem, ngl, nq, wnq, xgl, xnq):
    # intma
    # coord
    # nelem is the number of elements
    # ngl is the number of LGL points
    # nq  is the number of quadrature points
    # wnq are the lgl weights returned by lgl_gen(P) (returns xlgl, wlgl)
    # c is either child 1 or child 2

    #get psi1 and psi 2
    psi1, dpsi1 = S_psi(ngl, nq, xgl, xnq, 1)
    psi2, dpsi2 = S_psi(ngl, nq, xgl, xnq, 2)
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)

    #initialize element S matrix
    S1 = np.zeros((nelem,ngl,ngl))
    S2 = np.zeros((nelem,ngl,ngl))
    x = np.zeros(ngl)

    print(f'\n\n S Matrix Info:')

    for e in range(nelem):
        print(f'element: {e}')

        #store coordinates
        for i in range(ngl):
            I = int(intma[i][e])
            x[i] = coord[I]
            print(f'i = {i}, I = {I}, x_{i} = {x[i]}')

       

        #unstructured
        dx = x[-1]-x[0] #check this --> sould use len(x)?
        jac = dx/2.0

        #Do LGL integration
        for k in range(nq):
            wk=wnq[k]*jac

            for i in range(ngl):
                # h_i1=psi1[i][k]
                # h_i2=psi2[i][k]
                h_i1=psi[i][k]
                h_i2=psi[i][k]
                for j in range(ngl):
                    h_j1=psi1[j][k]
                    h_j2=psi2[j][k]
                    S1[e][i][j] = S1[e][i][j] + wk*h_i1*h_j1
                    S2[e][i][j] = S2[e][i][j] + wk*h_i2*h_j2

#             if (e == 0 and k==nq-1):
#                 print(f'e:{e}\n k:{k}\n i:{i}\n j:{j}\n')
#                 print(e_mass[e][i][j])

    ''' returns two 3D arrays with dimensions (nelem, ngl,ngl) '''
    return S1, S2


def create_scatters(M, S1, S2):
  ''' M, S1, and S2, are 3D arrays with dimentions (nelem, ngl, ngl). '''
  nelem = M.shape[0]
  ngl = M.shape[1]
  PS1 = np.empty((nelem,ngl,ngl))
  PS2 = np.empty((nelem,ngl,ngl))
#   PS1 = np.empty((nelem))
#   PS2 = np.empty((nelem))
  for e in range(nelem):
    print(f'element e: {e}')
    Minv = np.linalg.inv(M[e])  # invert the mass matrix for element e.
    print(f'Scattering routine. Mass matrix has shape {np.shape(M)}')
    print(f'specific dimension access test: tesing M{e}: M{e}, has shape {np.shape(M[e])} ')
    print(f'Scattering routine. S1 matrix has shape {np.shape(S1)}')
    print(f'specific dimension access test: tesing S1{e}: S1{e}, has shape {np.shape(S1[e])} ')
  
    PS1[e] = np.matmul(Minv, S1[e])
    PS2[e] = np.matmul(Minv, S2[e])
  print(f'Scattter Matrices complete. PS1 and PS2 have dimensions: {np.shape(PS1)},\n and {np.shape(PS2)}')
  return PS1, PS2

def create_gather(M, S1, S2):
  nelem = M.shape[0]
  ngl = M.shape[1]
  PG1 = np.empty((nelem,ngl,ngl))
  PG2 = np.empty((nelem,ngl,ngl))
  s = 0.5
  for e in range(nelem):
    Minv = np.linalg.inv(M[e]) # invert the mass matrix for element e.
    PG1[e] = s*np.matmul(Minv, S1[e].T)
    PG2[e] = s*np.matmul(Minv, S2[e].T)

  print(f'Gather Matrices complete. PG1 and PG2 have dimensions: {np.shape(PG1)},\n and {np.shape(PG2)}')
  return PG1, PG2