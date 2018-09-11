


BNCNN["S", n_] := NetChain[{
	ConvolutionLayer[64, {3, 3}, "Biases" -> None, "PaddingSize" -> 1],
	BatchNormalizationLayer[],
	Ramp
}];
DnCNN["S"] = NetChain[Flatten@{
	ConvolutionLayer[64, {3, 3}, "PaddingSize" -> 1, "Biases" -> None],
	Ramp,
	Table[BNCNN["S", n], {n, 1, 15}],
	ConvolutionLayer[3, {3, 3}, "PaddingSize" -> 1]
},
	"Input" -> {3, 640, 360}
] // NetInitialize;



BNCNN["B", n_] := NetChain[{
	BatchNormalizationLayer[],
	Ramp,
	ConvolutionLayer[64, {3, 3}, "Biases" -> None, "PaddingSize" -> 1]
}];
DnCNN["B"] = NetChain[Flatten@{
	NetChain[{
		ConvolutionLayer[64, {3, 3}, "Biases" -> None, "PaddingSize" -> 1],
		Ramp,
		ConvolutionLayer[64, {3, 3}, "Biases" -> None, "PaddingSize" -> 1]
	}],
	Table[BNCNN["B", n], {n, 1, 18}]
},
	"Input" -> {3, 640, 360}
] // NetInitialize;