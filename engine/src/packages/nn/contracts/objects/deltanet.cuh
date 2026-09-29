#ifndef SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_DELTANET_CUH
#define SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_DELTANET_CUH

/* This file needs nothing: every name in it is its own. */
#define NN__DELTANET__FAULT_BOUNDS 0x444E424Eull   /* "DNBN" — a shape past what the arrays hold */

/* The conv's two constants, and they are one apart by arithmetic rather than by coincidence — ▶ the
 * block above `conv_step`: a window of `TAPS - 1` is the smallest that yields exactly one output. */
#define NN__DELTANET__CONV_TAPS    4u    /* `linear_conv_kernel_dim` — `conv1d.weight` is [ch x 4] */
#define NN__DELTANET__CONV_WINDOW  3u    /* the retained columns: t-3, t-2, t-1                    */

#endif /* SILVANN__PACKAGES_NN_CONTRACTS_OBJECTS_DELTANET_CUH */
