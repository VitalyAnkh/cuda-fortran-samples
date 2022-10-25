	.file	"avx.c"
	.text
	.align	16
	.globl	v4half2float
	.globl  v4half2float_
v4half2float:
v4half2float_:
	vmovq	(%rsi), %xmm0
	vcvtph2ps  %xmm0, %xmm0
	vmovups	%xmm0, (%rdi)
	ret
	.type	v4half2float,@function
	.size	v4half2float,.-v4half2float
__v4half2floatEND:

	.text
	.align	16
	.globl	v8half2float
	.globl	v8half2float_
v8half2float:
v8half2float_:
	vmovdqu	(%rsi), %xmm0
	vcvtph2ps  %xmm0, %ymm0
	vmovups	%ymm0, (%rdi)
	ret
	.type	v8half2float,@function
	.size	v8half2float,.-v8half2float
__v8half2floatEND:

	.text
	.align	16
	.globl	v4float2half
	.globl	v4float2half_
v4float2half:
v4float2half_:
	vmovups	(%rsi), %xmm0
	vcvtps2ph  $4, %xmm0, %xmm0
	vmovq	%xmm0, (%rdi)
	ret
	.type	v4float2half,@function
	.size	v4float2half,.-v4float2half
__v4float2half:

	.text
	.align	16
	.globl	v8float2half
	.globl	v8float2half_
v8float2half:
v8float2half_:
	vmovups	(%rsi), %ymm0
	vcvtps2ph  $4, %ymm0, %xmm0
	vmovdqu	%xmm0, (%rdi)
	ret
	.type	v8float2half,@function
	.size	v8float2half,.-v8float2half
__v8float2half:

	.ident	"PGC 18.10-1"
