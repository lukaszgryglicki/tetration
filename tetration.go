package main

import (
	"math"
	"math/cmplx"
)

/*
High-accuracy complex tetration:
  F_b(h) with F(h+1)=b^{F(h)}, F(0)=1.

Regimes:
  (A) Shell–Thron interior (|lambda|<1): Schröder/Koenigs series + reversion + scaling by iterates.
  (B) Shell–Thron boundary (|lambda|≈1): parabolic -> use swordTrack gluing (robust near-neutral).
  (C) Outside Shell–Thron (repelling only): Kneser canonical via swordTrack gluing.

Public API:
  Tertrate(base, height complex128) complex128
  TetrationGrid(base complex128, rmin, imin, rmax, imax float64, nR, nI int) [][]complex128
*/

// ---------------- Utilities ----------------

func isBad(z complex128) bool {
	return math.IsNaN(real(z)) || math.IsNaN(imag(z)) ||
		math.IsInf(real(z), 0) || math.IsInf(imag(z), 0)
}
func cabs(z complex128) float64 { return cmplx.Abs(z) }

func safeLog(z complex128) complex128 {
	// principal branch
	return cmplx.Log(z)
}

// power with real exponent small helper
func cpow(a, b complex128) complex128 { return cmplx.Exp(b * cmplx.Log(a)) }

// ---------------- Lambert W (multi-branch) ----------------

// lambertWk solves W*exp(W)=z on branch k using Halley iterations with asymptotic seed.
func lambertWk(z complex128, k int) complex128 {
	if z == 0 {
		if k == 0 {
			return 0
		}
	}
	L1 := cmplx.Log(z) + complex(0, 2*math.Pi*float64(k))
	w := L1 - cmplx.Log(L1) // Fritsch seed
	const maxIt = 80
	const tol = 1e-15
	for i := 0; i < maxIt; i++ {
		e := cmplx.Exp(w)
		f := w*e - z
		den := e*(w+1) - (w+2)*f/(2*(w+1))
		if den == 0 {
			break
		}
		dw := f / den
		w2 := w - dw
		if cabs(dw) < tol*(1+cabs(w2)) {
			return w2
		}
		w = w2
	}
	return w
}

// ---------------- Formal power series helpers ----------------

type series struct {
	c []complex128 // c[n] z^n, length = N+1
}

func newSeries(N int) series {
	return series{c: make([]complex128, N+1)}
}
func (s series) deg() int { return len(s.c) - 1 }

func (s series) add(t series) series {
	N := s.deg()
	if t.deg() < N {
		N = t.deg()
	}
	u := newSeries(max(s.deg(), t.deg()))
	copy(u.c, s.c)
	for i := 0; i <= t.deg(); i++ {
		u.c[i] += t.c[i]
	}
	return u
}
func (s series) scale(a complex128) series {
	u := newSeries(s.deg())
	for i := range s.c {
		u.c[i] = a * s.c[i]
	}
	return u
}
func (s series) mul(t series) series {
	N := min(s.deg()+t.deg(), max(s.deg(), t.deg()))
	u := newSeries(N)
	for i := 0; i <= s.deg(); i++ {
		ci := s.c[i]
		if ci == 0 {
			continue
		}
		for j := 0; j <= t.deg() && i+j <= N; j++ {
			u.c[i+j] += ci * t.c[j]
		}
	}
	return u
}
func (s series) powInt(m, N int) series {
	// s^m truncated to degree N
	if m == 0 {
		u := newSeries(N)
		u.c[0] = 1
		return u
	}
	u := s
	for k := 1; k < m; k++ {
		u = u.mul(s)
		if u.deg() > N {
			u.c = u.c[:N+1]
		}
	}
	if u.deg() > N {
		u.c = u.c[:N+1]
	}
	return u
}
func (s series) compose(t series) series {
	// s(t(z)) truncated to degree min
	N := min(s.deg(), t.deg()) + (s.deg()-1)*t.deg()
	// cap by a safe length (we’ll truncate to len(t) eventually)
	M := min(N, max(0, len(s.c)+len(t.c)-2))
	u := newSeries(min(M, max(s.deg(), t.deg())))
	u.c[0] = s.c[0]
	// Precompute t^k
	pows := make([]series, s.deg()+1)
	pows[0] = unitSeries(u.deg())
	pows[1] = t
	for k := 2; k <= s.deg(); k++ {
		pows[k] = pows[k-1].mul(t)
		if pows[k].deg() > u.deg() {
			pows[k].c = pows[k].c[:u.deg()+1]
		}
	}
	for k := 1; k <= s.deg(); k++ {
		u = u.add(pows[k].scale(s.c[k]))
	}
	if u.deg() > t.deg() {
		u.c = u.c[:t.deg()+1]
	}
	return u
}
func unitSeries(N int) series {
	u := newSeries(N)
	u.c[0] = 1
	return u
}
func (s series) deriv() series {
	if len(s.c) <= 1 {
		return newSeries(0)
	}
	u := newSeries(s.deg() - 1)
	for n := 1; n < len(s.c); n++ {
		u.c[n-1] = complex(float64(n), 0) * s.c[n]
	}
	return u
}
func (s series) eval(z complex128) complex128 {
	// Horner
	N := s.deg()
	sum := complex(0, 0)
	for n := N; n >= 0; n-- {
		sum = sum*z + s.c[n]
	}
	return sum
}
func (s series) truncate(N int) series {
	if s.deg() <= N {
		return s
	}
	u := newSeries(N)
	copy(u.c, s.c[:N+1])
	return u
}

// reversion t such that s(t(w)) = w, assuming s(0)=0, s'(0)=1.
// If s(w) = w + sum_{k>=2} a_k w^k, we compute t(w) = w + sum_{n>=2} b_n w^n
// with b_n = - sum_{k=2}^n a_k * [w^n] (t(w))^k.
func reversion(s series) series {
	N := s.deg()
	t := newSeries(N)
	t.c[0] = 0
	t.c[1] = 1
	for n := 2; n <= N; n++ {
		var sum complex128
		// Use only the known part of t up to degree n-1
		tTrunc := t.truncate(n - 1)
		for k := 2; k <= n; k++ {
			a := s.c[k]
			if a == 0 {
				continue
			}
			p := powSeriesCoeff(tTrunc, k, n) // coeff of w^n in (t(w))^k
			sum += a * p
		}
		t.c[n] = -sum
	}
	return t
}
func powSeriesCoeff(t series, k, n int) complex128 {
	// coefficient of w^n in (t(w))^k
	// naive convolution by repeated multiplication truncated to n
	p := unitSeries(n)
	for i := 0; i < k; i++ {
		p = p.mul(t).truncate(n)
	}
	return p.c[n]
}

func min(a, b int) int { if a<b { return a }; return b }
func max(a, b int) int { if a>b { return a }; return b }

// ---------------- Schröder/Koenigs around an attracting fixed point ----------------

// buildKoenigs returns φ(z)=z+∑_{n≥2} a_n z^n satisfying φ(g(z))=λ φ(z)
// for g(z)=f(L+z)-L = L (e^{c z}-1). Also returns inverse ψ=φ^{-1}.
type koenigs struct {
	phi   series // around 0; degree N
	phiInv series
	L, c, lambda complex128
}

func buildKoenigs(L, c, lambda complex128, N int) koenigs {
	// g(z)=Σ_{m≥1} g_m z^m with g_m = L c^m / m!
	g := newSeries(N)
	fact := 1.0
	for m := 1; m <= N; m++ {
		fact *= float64(m)
		g.c[m] = L * cmplx.Pow(c, complex(float64(m), 0)) / complex(fact, 0)
	}
	// Precompute g^m
	gpow := make([]series, N+1)
	gpow[1] = g
	for m := 2; m <= N; m++ {
		gpow[m] = gpow[m-1].mul(g).truncate(N)
	}
	phi := newSeries(N)
	phi.c[0] = 0
	phi.c[1] = 1
	// Recurrence: a_n = (g_n + sum_{m=2}^{n-1} a_m*[z^n]g^m) / (λ - λ^n)
	for n := 2; n <= N; n++ {
		gn := g.c[n]
		sum := complex(0, 0)
		for m := 2; m <= n-1; m++ {
			a := phi.c[m]
			if a != 0 {
				sum += a * gpow[m].c[n]
			}
		}
		den := lambda - cmplx.Pow(lambda, complex(float64(n), 0))
		phi.c[n] = (gn + sum) / den
	}
	return koenigs{
		phi:    phi,
		phiInv: reversion(phi),
		L:      L, c: c, lambda: lambda,
	}
}

func (k koenigs) f(z complex128) complex128 { return cmplx.Exp(k.c * z) }

// psi1 = Koenigs(1) via limit λ^{-n}(f^n(1)-L)
func (k koenigs) psiAtOneOLD() complex128 {
	const maxN = 500
	const tol = 1e-15
	z := complex(1, 0)
	scale := complex(1, 0)
	prev := complex(math.Inf(1), 0)
	for n := 0; n < maxN; n++ {
		if n > 0 {
			z = k.f(z)
			scale *= k.lambda
		}
		w := (z - k.L) / scale
		if n > 5 && cabs(w-prev) < tol*(1+cabs(w)) {
			return w
		}
		prev = w
	}
	return prev
}

// Evaluate canonical tetration via Schröder in its basin (|lambda|<1, or near-boundary).
func tetrateSchroeder(base complex128, c, L, lambda, h complex128) complex128 {
	// LG:
	N := 96 // series order (can increase for harder bases)
	k := buildKoenigs(L, c, lambda, N)
	// K := k.psiAtOneOLD()
	// K := k.psiAtOneAttractingOLD()
	K := k.psiAbs(1)
	// target w = K * lambda^h
	w := K * cmplx.Exp(h*cmplx.Log(k.lambda))

	// scale down to within radius using n forward iterates after inversion
	// choose r ~ 0.20 in Koenigs-plane
	const r = 0.20
	wmag := cabs(w)
	lmag := cabs(k.lambda)
	n := 0
	if wmag > r && lmag > 0 {
		n = int(math.Ceil(math.Log(wmag/r) / math.Log(1.0/lmag)))
		if n < 0 { n = 0 }
		if n > 240 { n = 240 }
	}
	ws := w / cmplx.Pow(k.lambda, complex(float64(n), 0))
	// invert using series
	xi := k.phiInv.eval(ws)
	z := k.L + xi
	for i := 0; i < n; i++ {
		z = k.f(z)
	}
	return z
}

// ---------------- Sword-Track (Kneser canonical) ----------------

type fixedPoint struct {
	L, lambda complex128
	branch    int
}

// Build Koenigs charts at two repelling fixed points and glue across ℝ with Fourier maps ρ±.
type swordTrackSolver struct {
	// base data
	b, c complex128
	// fixed points
	up, dn fixedPoint
	// local charts
	upK, dnK koenigs
	// scaling ψ(1)
	upPsi1, dnPsi1 complex128
	// Fourier coefficients for ρ±(h)=h+Σ c_k e^{±2π i k h}
	M        int
	topC     []complex128
	botC     []complex128
	maxIt    int
	tol      float64
	samples  int
}

func newSwordTrack(b complex128, fps []fixedPoint) *swordTrackSolver {
	// choose conjugate-like pair with ±Im
	// pick two with smallest |Im| but opposite signs if possible
	var up, dn fixedPoint
	up = fps[0]
	dn = fps[1%len(fps)]
	for _, f := range fps {
		if imag(f.L) > 0 && (imag(up.L) <= 0 || imag(f.L) < imag(up.L)) {
			up = f
		}
		if imag(f.L) < 0 && (imag(dn.L) >= 0 || -imag(f.L) < -imag(dn.L)) {
			dn = f
		}
	}
	c := safeLog(b)
	N := 44 // series order per chart
	upK := buildKoenigs(up.L, c, up.lambda, N)
	dnK := buildKoenigs(dn.L, c, dn.lambda, N)
	s := &swordTrackSolver{
		b: b, c: c,
		up: up, dn: dn,
		upK: upK, dnK: dnK,
		// upPsi1: upK.psiAtOneOLD(),
		// dnPsi1: dnK.psiAtOneOLD(),
		// upPsi1: upK.psiAtOneRepellingOLD(),
		// dnPsi1: dnK.psiAtOneRepellingOLD(),
		upPsi1: upK.psiAbs(1),
		dnPsi1: dnK.psiAbs(1),
		// LG: was 14
		M: 24, // modes (increase for extreme accuracy)
		maxIt: 18,
		tol: 5e-13,
		// LG: was 8*14
		samples: 12*24, // oversampled real grid in [0,1)
	}
	s.topC = make([]complex128, s.M+1) // index 1..M used
	s.botC = make([]complex128, s.M+1)
	return s
}

// ρ maps
func (s *swordTrackSolver) rhoTop(h complex128) complex128 {
	z := h
	for k := 1; k <= s.M; k++ {
		if s.topC[k] != 0 {
			z += s.topC[k] * cmplx.Exp(complex(0, 2*math.Pi*float64(k))*h)
		}
	}
	return z
}
func (s *swordTrackSolver) rhoBot(h complex128) complex128 {
	z := h
	for k := 1; k <= s.M; k++ {
		if s.botC[k] != 0 {
			z += s.botC[k] * cmplx.Exp(complex(0, -2*math.Pi*float64(k))*h)
		}
	}
	return z
}

// Evaluate chart T(h)=L + φ^{-1}( ψ(1) * λ^{h} ) with scaling by iterates (|λ|>1 -> pull back)
func chartEval(k koenigs, psi1 complex128, h complex128) complex128 {
	l := k.lambda
	w := psi1 * cmplx.Exp(h*cmplx.Log(l))
	// scale: for |l|>1 choose n so that |w * l^{-n}| small
	const r = 0.20
	wmag := cabs(w)
	lmag := cabs(l)
	n := 0
	if wmag > r && lmag > 1 {
		n = int(math.Ceil(math.Log(wmag/r) / math.Log(lmag)))
		if n < 0 { n = 0 }
		if n > 240 { n = 240 }
	}
	ws := w / cmplx.Pow(l, complex(float64(n), 0))
	xi := k.phiInv.eval(ws)
	z := k.L + xi
	for i := 0; i < n; i++ {
		z = cmplx.Exp(k.c * z)
	}
	return z
}

func (s *swordTrackSolver) Ttop(h complex128) complex128 {
	return chartEval(s.upK, s.upPsi1, s.rhoTop(h))
}
func (s *swordTrackSolver) Tbot(h complex128) complex128 {
	return chartEval(s.dnK, s.dnPsi1, s.rhoBot(h))
}

// complex-step derivative
func derivComplexStep(F func(complex128) complex128, x complex128) complex128 {
	eps := 1e-8
	return imagUnitInv * (F(x+imagUnit*complex(eps,0)) - F(x))
}
var imagUnit = complex(0,1)
var imagUnitInv = complex(0,-1)

// Least-squares solve of A u = b for complex A (normal equations, naive)
// A: S x P, b: S. Build G = A^* A (P x P), rhs = A^* b and do Gaussian elim.
func solveNormalEq(A [][]complex128, b []complex128) []complex128 {
	S := len(A)
	if S == 0 { return nil }
	P := len(A[0])
	G := make([][]complex128, P)
	for i := range G {
		G[i] = make([]complex128, P)
	}
	rhs := make([]complex128, P)
	// G= A^* A, rhs = A^* b
	for s := 0; s < S; s++ {
		for j := 0; j < P; j++ {
			aj := cmplx.Conj(A[s][j])
			rhs[j] += aj * b[s]
			for k := 0; k < P; k++ {
				G[j][k] += aj * A[s][k]
			}
		}
	}
	// Gaussian elimination
	u := make([]complex128, P)
	// forward
	for i := 0; i < P; i++ {
		// pivot
		p := i
		best := cabs(G[i][i])
		for r := i + 1; r < P; r++ {
			if cabs(G[r][i]) > best {
				best = cabs(G[r][i]); p = r
			}
		}
		if p != i {
			G[i], G[p] = G[p], G[i]
			rhs[i], rhs[p] = rhs[p], rhs[i]
		}
		pi := G[i][i]
		if pi == 0 {
			continue
		}
		// normalize
		for k := i; k < P; k++ {
			G[i][k] /= pi
		}
		rhs[i] /= pi
		// eliminate
		for r := i + 1; r < P; r++ {
			f := G[r][i]
			if f == 0 { continue }
			for k := i; k < P; k++ {
				G[r][k] -= f * G[i][k]
			}
			rhs[r] -= f * rhs[i]
		}
	}
	// back-substitution
	for i := P - 1; i >= 0; i-- {
		sum := rhs[i]
		for k := i + 1; k < P; k++ {
			sum -= G[i][k] * u[k]
		}
		if G[i][i] != 0 {
			u[i] = sum / G[i][i]
		} else {
			u[i] = 0
		}
	}
	return u
}

// Iterate Fourier gluing so that Ttop(ρ_top(x)) == Tbot(ρ_bot(x)) on real axis (x in [0,1))
func (s *swordTrackSolver) solve() {
	S := s.samples
	X := make([]float64, S)
	for i := 0; i < S; i++ {
		X[i] = float64(i) / float64(S) // [0,1)
	}
	P := 2 * s.M // unknowns: M top + M bottom
	// iteration
	for it := 0; it < s.maxIt; it++ {
		// Build residual D and Jacobian-projected design matrix
		D := make([]complex128, S)
		A := make([][]complex128, S)
		for n := 0; n < S; n++ {
			x := X[n]
			xx := complex(x, 0)
			// residual at x
			F1 := s.Ttop(xx)
			F2 := s.Tbot(xx)
			D[n] = F1 - F2
			// Jacobians wrt rho (approx via complex-step)
			J1 := derivComplexStep(s.Ttop, xx)
			J2 := derivComplexStep(s.Tbot, xx)
			// row: [ J1*e^{+2πikx} |  -J2*e^{-2πikx} ]
			row := make([]complex128, P)
			for k := 1; k <= s.M; k++ {
				row[k-1] = J1 * cmplx.Exp(complex(0, 2*math.Pi*float64(k))*xx)
				row[s.M+(k-1)] = -J2 * cmplx.Exp(complex(0, -2*math.Pi*float64(k))*xx)
			}
			A[n] = row
		}
		// Solve least squares A u ≈ -D
		u := solveNormalEq(A, negateVec(D))
		// Update
		for k := 1; k <= s.M; k++ {
			s.topC[k] += u[k-1]
			s.botC[k] += u[s.M+(k-1)]
		}
		// check residual
		nrm := 0.0
		for n := 0; n < S; n++ {
			nrm += cabs(D[n]) * cabs(D[n])
		}
		nrm = math.Sqrt(nrm / float64(S))
		if nrm < s.tol {
			break
		}
	}
}

func negateVec(v []complex128) []complex128 {
	u := make([]complex128, len(v))
	for i := range v {
		u[i] = -v[i]
	}
	return u
}

// Public evaluation (canonical Kneser branch)
func (s *swordTrackSolver) eval(h complex128) complex128 {
	if imag(h) > 0 {
		return s.Ttop(h)
	}
	if imag(h) < 0 {
		return s.Tbot(h)
	}
	// on real axis they coincide; to be safe, average
	return 0.5 * (s.Ttop(h) + s.Tbot(h))
}

// choose the logarithm branch so that (Log(z)+2πi m)/c is closest to L
func logNearFixed(z, c, L complex128) complex128 {
	base := cmplx.Log(z)
	best := base
	bestDist := cmplx.Abs(best/c - L)
	// try a few neighboring 2πi shifts
	for m := -6; m <= 6; m++ {
		if m == 0 { continue }
		cand := base + complex(0, 2*math.Pi*float64(m))
		d := cmplx.Abs(cand/c - L)
		if d < bestDist {
			best = cand
			bestDist = d
		}
	}
	return best
}

// For |λ|<1 (attracting): ψ(1) = lim_{n→∞} λ^{-n} (f^n(1) - L),
// computed robustly without letting λ^n underflow to 0.
func (k koenigs) psiAtOneAttractingOLD() complex128 {
	const maxN = 300
	const tol  = 1e-14
	const smin = 1e-300

	z := complex(1, 0)
	scale := complex(1, 0) // λ^n
	var prev complex128
	havePrev := false

	for n := 0; n < maxN; n++ {
		if n > 0 {
			z = k.f(z)      // forward iterate
			scale *= k.lambda
			if cmplx.Abs(scale) < smin {
				// scale underflow imminent—return last stable value
				if havePrev { return prev }
				break
			}
		}
		w := (z - k.L) / scale // λ^{-n}(f^n(1)-L)
		if havePrev && cmplx.Abs(w-prev) < tol*(1+cmplx.Abs(w)) {
			return w
		}
		prev = w
		havePrev = true
	}
	return prev
}

// For |λ|>1 (repelling): ψ(1) must be built from inverse iterates:
// ψ(1) = lim_{n→∞} λ^{n} ( f^{-n}(1) - L ).
func (k koenigs) psiAtOneRepellingOLD() complex128 {
	const maxN = 300
	const tol  = 1e-14
	const smax = 1e300

	z := complex(1, 0)
	scale := complex(1, 0) // λ^n
	var prev complex128
	havePrev := false

	for n := 0; n < maxN; n++ {
		if n > 0 {
			if z == 0 {
				return complex(math.NaN(), math.NaN())
			}
			// inverse iterate with branch chosen to approach L
			z = logNearFixed(z, k.c, k.L) / k.c
			scale *= k.lambda
			if cmplx.Abs(scale) > smax {
				// avoid overflow; return last stable
				if havePrev { return prev }
				break
			}
		}
		w := (z - k.L) * scale // λ^{n}(f^{-n}(1)-L)
		if havePrev && cmplx.Abs(w-prev) < tol*(1+cmplx.Abs(w)) {
			return w
		}
		prev = w
		havePrev = true
	}
	return prev
}

// psiAbs(z): compute ψ(z)=φ(z-L) continued by inverse iterates so that
// z_n approaches L and |z_n-L| is within φ's radius. Then ψ(z)=λ^n φ(z_n-L).
func (k koenigs) psiAbs(z complex128) complex128 {
	// radius from coefficients, with safety margin
	R := seriesRadius(k.phi)
	target := 0.6 * R
	if target < 1e-6 {
		target = 1e-6
	}
	const maxN = 120 // number of pull-back steps
	cur := z
	n := 0

	for n = 0; n < maxN; n++ {
		if cmplx.Abs(cur-k.L) <= target {
			val := k.phi.eval(cur - k.L)
			return cmplx.Pow(k.lambda, complex(float64(n), 0)) * val
		}

		// Avoid the "1 → 0" trap: if cur≈1, do NOT use m=0; pick best nonzero branch.
		if cmplx.Abs(cur-1) < 1e-14 {
			best := complex(0, 0)
			bestDist := math.Inf(1)
			for m := -12; m <= 12; m++ {
				if m == 0 {
					continue
				}
				cand := complex(0, 2*math.Pi*float64(m))
				next := cand / k.c
				d := cmplx.Abs(next - k.L)
				if d < bestDist {
					bestDist = d
					best = cand
				}
			}
			// pull back using the chosen nonzero branch
			cur = best / k.c
			continue
		}

		// Generic branch choice: pick the log branch that moves toward L
		cur = logNearFixed(cur, k.c, k.L) / k.c
	}

	// Fallback: if we never got sufficiently close, evaluate series anyway (may be poor).
	return k.phi.eval(z - k.L)
}


// rough radius of convergence estimate for a power series s(z)=z+Σ_{n>=2} a_n z^n
func seriesRadius(s series) float64 {
	R := math.Inf(1)
	for n := 2; n <= s.deg(); n++ {
		an := s.c[n]
		if an == 0 {
			continue
		}
		r := math.Pow(cmplx.Abs(an), -1.0/float64(n))
		if r < R {
			R = r
		}
	}
	if math.IsInf(R, 0) || math.IsNaN(R) {
		return 1.0 // safe default
	}
	return R
}


// ---------------- Region detection and main API ----------------

func Tertrate(base, height complex128) complex128 {
	// Reject trivial/undefined bases
	if base == 0 || base == 1 || isBad(base) {
		return cmplx.NaN()
	}
	c := safeLog(base)
	// fixed points L_k = -W_k(-c)/c, lambda_k = -W_k(-c)
	branches := []int{0, -1, 1, -2, 2, -3, 3}
	var fps []fixedPoint
	for _, k := range branches {
		w := lambertWk(-c, k)
		if isBad(w) {
			continue
		}
		L := -w / c
		lmb := -w
		fps = append(fps, fixedPoint{L: L, lambda: lmb, branch: k})
	}
	if len(fps) == 0 {
		return cmplx.NaN()
	}
	// pick attracting if available
	var best *fixedPoint
	bestAbs := math.MaxFloat64
	for i := range fps {
		m := cabs(fps[i].lambda)
		if m < 1 && m < bestAbs {
			best = &fps[i]
			bestAbs = m
		}
	}
	if best != nil {
		// Schröder (interior)
		return tetrateSchroeder(base, c, best.L, best.lambda, height)
	}
	// near-parabolic boundary? (|lambda|-1 small)
	near := &fps[0]
	delta := math.Abs(cabs(near.lambda) - 1)
	for i := range fps {
		d := math.Abs(cabs(fps[i].lambda) - 1)
		if d < delta {
			delta = d; near = &fps[i]
		}
	}
	if delta < 3e-3 {
		// Treat with sword-track gluing (robust around parabolic)
		// Ensure we have at least two distinct fp's
		if len(fps) < 2 {
			return cmplx.NaN()
		}
		st := newSwordTrack(base, fps)
		// tighten for parabolic case
		// st.M = 18; st.samples = 10*st.M; st.tol = 2e-13; st.maxIt = 24
		// LG:
		st.M = 22; st.samples = 12*st.M; st.tol = 2e-13; st.maxIt = 28
		st.solve()
		return st.eval(height)
	}
	// Repelling-only (Kneser canonical) via sword-track
	st := newSwordTrack(base, fps)
	st.solve()
	return st.eval(height)
}

// Grid sampler: returns nR x nI matrix F[xi][yj] for h = (rmin + i*dr) + i (imin + j*di)
func TetrationGrid(base complex128, rmin, imin, rmax, imax float64, nR, nI int) [][]complex128 {
	if nR <= 0 || nI <= 0 {
		return nil
	}
	grid := make([][]complex128, nR)
	dr := (rmax - rmin) / float64(nR-1)
	di := (imax - imin) / float64(nI-1)
	for i := 0; i < nR; i++ {
		row := make([]complex128, nI)
		x := rmin + float64(i)*dr
		for j := 0; j < nI; j++ {
			y := imin + float64(j)*di
			h := complex(x, y)
			row[j] = Tertrate(base, h)
		}
		grid[i] = row
	}
	return grid
}

