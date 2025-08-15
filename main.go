package main

import (
	"fmt"
	"math"
)

func main() {
	// Examples
	fmt.Println("F_b(0) with b=1.3+0.2i =", Tertrate(1.3+0.2i, 0))
	fmt.Println("F_b(1) with b=1.3+0.2i =", Tertrate(1.3+0.2i, 1))
	fmt.Println("F_e(0.5) (canonical)   =", Tertrate(complex(math.E,0), 0.5))

	// Sample a rectangle in the h-plane (height plane):
	// base := complex(2, 0) // > e^(1/e): uses canonical Kneser via sword-track
	// grid := tetration.TetrationGrid(base, -1, -2, 2, 2, 81, 81) // 81x81 samples
	// _ = grid // use/save/visualize
	// Each grid[i][j] is complex128 = F_base(h_ij)
}

