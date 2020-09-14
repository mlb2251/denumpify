# denumpify

np.where((X==searched_values[:,None]).all(-1))[1] from

dist = numpy.linalg.norm(a-b)

A[np.arange(A.shape[0])[:,None],B] from

numpy.r_[True, a[1:] < a[:-1]] & numpy.r_[a[:-1] < a[1:], True] from

(A==B).all() from

np.nonzero(np.in1d(A,B))[0] from
np.searchsorted(A,B) from

a = np.array([5,4])[np.newaxis] from

test[:,0]

dists[abs(dists - r - dr/2.) <= dr/2.]  from
dists[(dists >= r) & (dists <= r+dr)]  from
dists[(np.where((dists >= r) & (dists <= r + dr)))]  from


b = np.repeat(a[:, :, np.newaxis], 3, axis=2) from

b = a.ravel() from
c = a.flatten() from

m,n = A.shape; np.take(A,B + n*np.arange(m)[:,None])

np.concatenate((a, b))
np.vstack((a, b))
a.dot(b)
B = a.T
numpy.hstack([a, a])
Z_new = Z.reshape(5,20,5,20).sum(axis=(1,3)) from
Z_new = np.einsum('ijkl->ik',Z.reshape(5,20,5,20)) from

unique, counts = numpy.unique(a, return_counts=True)
(-avgDists).argsort()[:n]
avgDists.argsort()[::-1][:n]
a[..., 0].flatten()
array([[1,2,3],]*3).transpose()
d += np.bincount(i,v,minlength=d.size).astype(d.dtype, copy=False)

c = (a < 3).astype(int)
a[np.arange(len(a)), [1,0,2]]
np.any(valeur <= 0.6)
np.repeat(data, 5)
