import numpy as np
import matplotlib.pyplot as plt
import itertools
import time


def rget_vec(vecs, idxs):
    if len(idxs) == 1:
        return vecs[idxs[0]]
        
    return rget_vec(vecs[idxs[0]], idxs[1:])

class Perlin:
    # https://flafla2.github.io/2014/08/09/perlinnoise.html
    # https://mrl.nyu.edu/~perlin/noise/
    def __init__(self, hash_size=256):
        self.hash_size = 256 # todo: find out how to make this value variable and still work
        self.perms = np.random.permutation(hash_size)
        self.p = np.concatenate([self.perms, self.perms])
        
        self.vec_tables = {}
    

    def noisend(self, *coords):
        _coords = list(map(lambda n: int(n) & (self.hash_size - 1), coords))
        coords = list(map(lambda n: n % 1, coords))
        vecs = list(map(self.fade, coords))
        
        
        combs = [c[::-1] for c in self.comb(len(coords))]
        pts = list(map(lambda c: self.hash(_coords, c), combs))
        
        grads = []
        for comb, pt in zip(combs, pts):
            cur_coords = [c - int(s) for c, s in zip(coords, comb)] # sub 1 in corresponding places
            grads.append(self.gradnd(pt, cur_coords))
            
        
        return self.rlerp(vecs[::-1], grads)
    
    def rlerp(self, weights, vals):
        if len(weights) == 1:
            return self.lerp(weights[0], vals[0], vals[1])
            
        n = len(vals)
        return self.lerp(
            weights[0], 
            self.rlerp(weights[1:], vals[:n//2]),
            self.rlerp(weights[1:], vals[n//2:]),    
            )
    
        
    def comb(self, depth=0):
        return list(itertools.product([0, 1], repeat=depth))
        
    def comb_posneg(self, depth=0):
        return list(itertools.product([-1, 1], repeat=depth))
        
    def hash(self, coords, seq):
        coords = coords + [0] # add dummy 0 as last add value
            
        c = coords[0]
        for i, s in enumerate(seq):
            c = self.p[c + int(s)] + coords[i+1]
            
        return c
        
    def create_vecs(self, n):
        combs = self.comb_posneg(n - 1)
        vecs = [
            c[:i] + (0,) + c[i:]
            for c in combs
            for i in range(n)
        ]
        np.random.shuffle(vecs)
        
        self.vec_tables[n] = vecs
        
    def gradnd(self, _hash, coords):
        if len(coords) not in self.vec_tables:
            self.create_vecs(len(coords))
            
        h = _hash % len(self.vec_tables[len(coords)])
        
        return np.dot(self.vec_tables[len(coords)][h], coords)


    def fade(self, t):
        # 6t^5 - 15t^4 + 10t^3
        return 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3
        
    def lerp(self, t, a, b):
        return a + t * (b - a)
        

if __name__ == "__main__":
    p = Perlin(256)

    st = time.time()
    grads2 = [[p.noisend(x/10, y/10, 0) for x in range(10*16)] for y in range(10*16)]
    print(time.time() - st)

    
    plt.imshow(grads2)
    plt.show()
    