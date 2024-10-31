import numpy as np
from scipy.linalg import subspace_angles
from scipy import linalg
import torch

@torch.no_grad()
def torch_gs_new(X):
    return torch.nn.Parameter(torch.from_numpy(gs(X.detach().cpu().numpy())))

@torch.no_grad()
def torch_gs(X, row_vecs=True, norm=True):
    U, S, V = torch.linalg.svd(X, full_matrices=False)

    # Reconstruct orthonormalized tensor
    orthonormal_tensor = torch.matmul(U, V)

    return orthonormal_tensor

# https://gist.github.com/iizukak/1287876?permalink_comment_id=1348649#gistcomment-1348649
def gs(X, row_vecs=True, norm = True):
    if not row_vecs:
        X = X.T
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
        # Y = Y / np.linalg.norm(Y, axis=1, keepdims=True)
    if row_vecs:
        return Y
    else:
        return Y.T

def get_principal_angles_samedim(A, B):
    new_A = gs(A.astype(np.longdouble), row_vecs=False)
    new_B = gs(B.astype(np.longdouble), row_vecs=False)

    mat = new_A.T @ new_B
    mat = mat / np.linalg.norm(mat)
    _, singular_values, _ = np.linalg.svd(mat)
    return np.arccos(singular_values.astype(np.longdouble))


def grassmann_distance_samedim(principal_angles):
    distance = 0
    for i in range(len(principal_angles)):
        distance += principal_angles[i]**2
    
    distance = distance ** 0.5
    return distance

def get_principal_angles_differentdim(A, B):
    pass

def grassmann_distance_differentdim(principal_angles):
    pass

def grassmann_distance(A, B):
    same_dim = A.shape[1] == B.shape[1]
    #principal_angles = get_principal_angles_samedim(A, B) if same_dim else get_principal_angles_differentdim(A, B)
    new_A = A / np.linalg.norm(A)
    new_B = B / np.linalg.norm(B)
    principal_angles = subspace_angles(new_A, new_B)
    # distance = grassmann_distance_samedim(principal_angles) if same_dim else grassmann_distance_differentdim(principal_angles)
    distance = grassmann_distance_samedim(principal_angles)
    return distance

def reconstruction_error(Uhat,U):
    """
    Args:
        Uhat:
            subspace estimates

        U:
            true subspace

    Return:
        reconstruction error
    """
    return linalg.norm(U - Uhat @ (Uhat.T @ U),ord='fro')**2

def x2states(x, d):
    x_size = x.shape
    t_count = x_size[1]
    P = []
    q = []
    r = []
    S = []
    z = []

    for i in range(t_count):
        s = vec2state(x[:, i], d)
        P.append(s[0])
        q.append(s[1])
        r.append(s[2])
        S.append(s[3])
        z.append(s[4])
    
    return np.array(P), np.array(q), np.array(r), np.array(S), np.array(z)

def state_to_vec(P, q, r, S, z, d):
    zz = np.zeros((int((d + 1) * d / 2), 1))

    k = 0
    for i in range(d):
        for j in range(i+1):
            zz[k] = S[i, j]
            k += 1
    P_T = P.T
    v = [P_T.flatten(), zz.flatten(), q.flatten(), np.array([z]), r.flatten()]
    return np.concatenate(v)

def new_state_to_vec(P, Q, R, S, Z, d):
    return np.hstack([P.flatten(), Q.flatten(), R.flatten(), S.flatten(), Z.flatten()])

def new_x2states(vec, d):
    vec = vec.T
    P = vec[:, :d**2].reshape(-1, d, d)
    Q = vec[:, d**2:2*d**2].reshape(-1, d, d)
    R = vec[:, 2*d**2:3*d**2].reshape(-1, d, d)
    S = vec[:, 3*d**2:4*d**2].reshape(-1, d, d)
    Z = vec[:, 4*d**2:5*d**2].reshape(-1, d, d)

    return P, Q, R, S, Z

def new_vec_to_state(vec, d):
    P = vec[:d**2].reshape(d, d)
    Q = vec[d**2:2*d**2].reshape(d, d)
    R = vec[2*d**2:3*d**2].reshape(d, d)
    S = vec[3*d**2:4*d**2].reshape(d, d)
    Z = vec[4*d**2:5*d**2].reshape(d, d)

    return P, Q, R, S, Z


def convert_states(states, d):
    shape = states.shape
    t = shape[1]
    P = []
    q = []
    r = []
    S = []
    z = []
    for i in range(t):
        s = vec2state(states[:, i], d)
        P.append(s[0])
        q.append(s[1])
        r.append(s[2])
        S.append(s[3])
        z.append(s[4])

    return np.array(P), np.array(q), np.array(r), np.array(S), np.array(z)

def vec2state(v, d):
    i = 0

    P = np.reshape(v[i:i+d*d], (d, d))
    P = P.T
    i = i + d*d

    S = np.zeros((d, d))

    k = i
    for I in range(d):
        for J in range(I+1):
            S[I, J] = v[k]
            S[J, I] = v[k]
            k += 1
    
    i = int(i + (d+1)*d / 2)

    q = v[i:i+d]
    i = i+d
    z = v[i]
    i += 1
    r = v[i:i+d]

    return P, q, r, S, z