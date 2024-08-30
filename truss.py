# Importing numpy for matrices and linear equation solving
import numpy as np
# Importing matplotlib for making plots
import matplotlib.pyplot as plt

# Young modulus
E = 210e9  # Pa
# gravitational acceleration
g = 9.81  # m/s^2
# density
rho = 7850  # kg/m^3
# crosse section area
A = 18e-3  # m^2

# List of points
points = np.array([
    (0,0),(1,0),(2,0),(3,0),
    (4,0),(0.5,1),(1.5,1),(2.5,1),
    (3.5,1)
])

# Defining a new type, consisting of two point indexes and a cross-section area
dt = np.dtype([("i1",np.int32),("i2",np.int32),("A",np.float64)])
# List of elements
elements = np.array([
    (0,1,A),(1,2,A),(2,3,A),(3,4,A),
    (5,6,A),(6,7,A),(7,8,A),(0,5,A),
    (5,1,A),(1,6,A),(6,2,A),(2,7,A),
    (7,3,A),(3,8,A),(8,4,A)
],dtype=dt)

# Function for plotting the beams
def plot_beams(p, e):
    for el in e:
        plt.plot(
            [p[el['i1'],0],p[el['i2'],0]],
            [p[el['i1'],1],p[el['i2'],1]],
        'k-')
    plt.plot(p[:,0], p[:,1],"o")
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()
plot_beams(points, elements)


# We have the same number of DOFs as we have points x 2
dofs = points.size

# We initialize the stiffness matrix and right-hand-side vector with zeros
S = np.zeros((dofs, dofs))
RHS = np.zeros(dofs)

# We loop over elements and assemble the matrix
for el in elements:
    # Indexes of local DOFs
    local_dof = np.array([2*el['i1'], 2*el['i1']+1, 2*el['i2'], 2*el['i2']+1])
    # Constructing vector along the beam
    n = points[el['i2']] - points[el['i1']]
    L = np.sqrt(n.dot(n))
    n = n / L
    N = np.array([[-n[0], -n[1], n[0], n[1]]])
    # Local stiffness matrix
    local_S = np.dot(N.T, np.dot(A*E/L, N))
    # Adding the local stiffness matrix to the global
    S[np.ix_(local_dof, local_dof)] += local_S

# Plotting the matrix
plt.imshow(S)
plt.show()

# Testing deformations that should give zero forces:
m = points.shape[0]
# 1. move everything in X
x = np.column_stack((np.ones(m),np.zeros(m))).reshape(dofs)
np.dot(S, x)
# 2. move everything in Y
x = np.column_stack((np.zeros(m),np.ones(m))).reshape(dofs)
np.dot(S, x)
# 3. rotate
x = np.column_stack((-points[:,1],points[:,0])).reshape(dofs)
np.dot(S, x)

# Adding load
to_load = np.array([2*2+1])
load = -1e3 * g  # 1 tonne
RHS[to_load] = load

# Adding supports
to_fix = np.array([2*0+0, 2*0+1, 2*4+0, 2*4+1])
S[to_fix, :] = np.eye(dofs)[to_fix, :]
RHS[to_fix] = 0

# Solving the system
x = np.linalg.solve(S, RHS)

# Reshaping the table from a vector to a table similar to 'points'
displacement = np.reshape(x, points.shape)

# Plotting exaggerated deformation
scale = 10000
plot_beams(points + scale * displacement, elements)
