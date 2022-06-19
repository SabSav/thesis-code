#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[NotebookDirectory[]];
readSample[nameFile_, nameQuantity_] := Import[nameFile, "RawJSON"][nameQuantity];
acrPlot[sample_, length_, label_, color_] := ListPlot[CorrelationFunction[sample, {length}], Filling -> Axis, 
													  PlotRange -> All,  PlotStyle-> color, PlotLabel-> label];
setLabels[nameQuantity_] := Import["theory-lT.json", "RawJSON"][nameQuantity];
generateCounts[nameFile_, nameQuantity_] := Sort[Tally[Import[nameFile, "RawJSON"][nameQuantity]]][[All, 2]];
sampleTheory[nsample_, nameQuantity_]:= Round[nsample * readSample["theory-lT.json", nameQuantity]];
statParamTheory[nsample_, nameQuantity_]:= (
	stdTheory = StandardDeviation[MultinomialDistribution[n= nsample, readSample["theory-lT.json", nameQuantity]]];
	meanTheory = Mean[MultinomialDistribution[n= nsample, readSample["theory-lT.json", nameQuantity]]];
	Return[{stdTheory, meanTheory}];
	)
	
stdSample[sample_, meanTheory_, stdTheory_] :=  
(
std = stdTheory;
sampleErrors = sample;
factor = sample;
For[i = 1, i <= Length[sample], i++,
	factor[[i]] = Round[Abs[sample[[i]] - meanTheory[[i]]] / stdTheory[[i]]];
	];
For[i = 1, i <= Length[sample], i++, 
	If[factor[[i]]> 1, std[[i]] = factor[[i]]* stdTheory[[i]] ];
	sampleErrors[[i]] = Around[sample[[i]], std[[i]]];
];
Return[sampleErrors];
)


engyMc = readSample["mc-lT.json","energy_sample"];
engyAlg1 = readSample["a1-lT.json","energy_sample"];
engyAlg2 = readSample["a2-lT.json","energy_sample"];
acrlMc = acrPlot[engyMc, 20, "Mc", Blue];
acrlAlg1 = acrPlot[engyAlg1, 20, "Alg1", Magenta];
acrlAlg2 = acrPlot[engyAlg2, 20, "Alg", Brown];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


labels = setLabels["energy_level"];
engyMc = generateCounts["mc-lT.json", "energy_sample"];
engyAlg1 = generateCounts["a1-lT.json", "energy_sample"];
engyAlg2 = generateCounts["a2-lT.json", "energy_sample"];
engyTheory = sampleTheory[Total[engyMc], "energy_probs"];
stats = statParamTheory[Total[engyMc], "energy_probs"];
stdTheory = stats[[1]];
meanTheory = stats[[2]];
engyMcErrors = stdSample[engyMc, meanTheory, stdTheory];
engyAlg1Errors = stdSample[engyAlg1, meanTheory, stdTheory];
engyAlg2Errors = stdSample[engyAlg2, meanTheory, stdTheory];
engyTheoryErrors = engyTheory;
For[i = 1, i <= Length[labels], i++,
	engyTheoryErrors[[i]] = Around[engyTheory[[i]], stdTheory[[i]]]
];

colors = ColorData[54];
baseStyle = Directive[Black, FontFamily->"Arial", FontSize->16];
frameStyle = Append[baseStyle, Thickness@0.003];
rotateLabel[label_]:=Style[Rotate[label,Pi/4],30,Bold,Opacity[0.2],FontFamily->"Helvetica"]
BarChart[{engyMcErrors, engyAlg1Errors, engyAlg2Errors, engyTheoryErrors}, 
ChartLabels->{Placed[{"Mc","Alg1", "Alg2", "Theory"},Center, rotateLabel],labels},
BaseStyle->baseStyle, FrameStyle->frameStyle, ImageSize-> Large, ChartLegends->labels]



magnMc = readSample["mc-lT.json","magnetization_sample"];
magnAlg1 = readSample["a1-lT.json","magnetization_sample"];
magnAlg2 = readSample["a2-lT.json","magnetization_sample"];
acrlMc = acrPlot[magnMc, 100, "Mc", Blue];
acrlAlg1 = acrPlot[magnAlg1, 100, "Alg1", Magenta];
acrlAlg2 = acrPlot[magnAlg2, 100, "Alg", Brown];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


labels = setLabels["magnetization_level"];
magnMc = generateCounts["mc-lT.json", "magnetization_sample"];
magnAlg1 = generateCounts["a1-lT.json", "magnetization_sample"];
magnAlg2 = generateCounts["a2-lT.json", "magnetization_sample"];
magnTheory = sampleTheory[Total[magnMc], "magnetization_probs"];
stats = statParamTheory[Total[magnMc], "magnetization_probs"];
stdTheory = stats[[1]];
meanTheory = stats[[2]];
magnMcErrors = stdSample[magnMc, meanTheory, stdTheory];
magnAlg1Errors = stdSample[magnAlg1, meanTheory, stdTheory];
magnAlg2Errors = stdSample[magnAlg2, meanTheory, stdTheory];
magnTheoryErrors = magnTheory;
For[i = 1, i <= Length[labels], i++,
	magnTheoryErrors[[i]] = Around[magnTheory[[i]], stdTheory[[i]]]
];

colors = ColorData[54];
baseStyle = Directive[Black, FontFamily->"Arial", FontSize->16];
frameStyle = Append[baseStyle, Thickness@0.003];
rotateLabel[label_]:=Style[Rotate[label,Pi/4],30,Bold,Opacity[0.2],FontFamily->"Helvetica"]
BarChart[{magnMcErrors, magnAlg1Errors, magnAlg2Errors, magnTheoryErrors}, 
ChartLabels->{Placed[{"Mc","Alg1", "Alg2", "Theory"},Center, rotateLabel],labels},
BaseStyle->baseStyle, FrameStyle->frameStyle, ImageSize-> Large, ChartLegends->labels]
