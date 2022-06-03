#!/usr/bin/env wolframscript
(* ::Package:: *)

(* ::Input:: *)
(**)


SetDirectory[NotebookDirectory[]];
engyMc = Import["mc-lT.json", "RawJSON"]["energy_sample"];
engyAlg1 = Import["a1-lT.json", "RawJSON"]["energy_sample"];
engyAlg2 = Import["a2-lT.json", "RawJSON"]["energy_sample"];
acrlMc = ListPlot[CorrelationFunction[engyMc, {20}], Filling -> Axis, PlotRange -> All, PlotLabel->"Mc"];
acrlAlg1 = ListPlot[CorrelationFunction[engyAlg1, {20}], Filling -> Axis, PlotRange -> All, PlotStyle->Magenta, PlotLabel->"Alg1"];
acrlAlg2 = ListPlot[CorrelationFunction[engyAlg2, {20}], Filling -> Axis, PlotRange -> All, PlotStyle->Brown, PlotLabel->"Alg2"];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


labels = Import["theory-lT.json", "RawJSON"]["energy_level"];
engyMc = Sort[Tally[Import["mc-lT.json", "RawJSON"]["energy_sample"]]][[All, 2]] ;
engyAlg1 = Sort[Tally[Import["a1-lT.json", "RawJSON"]["energy_sample"]]][[All, 2]];
engyAlg2 = Sort[Tally[Import["a2-lT.json", "RawJSON"]["energy_sample"]]][[All, 2]];
engyTheory = Round[Total[engyMc] * Import["theory-lT.json", "RawJSON"]["energy_probs"]];
stdTheory = StandardDeviation[MultinomialDistribution[n= Total[engyMc],Import["theory-lT.json", "RawJSON"]["energy_probs"]]];
meanTheory =  Mean[MultinomialDistribution[n= Total[engyMc],Import["theory-lT.json", "RawJSON"]["energy_probs"]]];
factorMc = Round[Abs[engyMc - meanTheory] / stdTheory];
stdMc = stdTheory;
engyMcErrors = engyMc;
For[i = 1, i <= Length[labels], i++,
	If[factorMc [[i]]> 1, stdMc[[i]] = factorMc[[i]]* stdTheory[[i]]];
	engyMcErrors[[i]] = Around[engyMc[[i]], stdMc[[i]]];
	]
factorAlg1 = Round[Abs[engyAlg1 - meanTheory] / stdTheory];
stdAlg1 = stdTheory;
engyAlg1Errors = engyAlg1;
For[i = 1, i <= Length[labels], i++, 
	If[factorAlg1[[i]]> 1, stdAlg1[[i]] = factorAlg1[[i]]* stdTheory[[i]] ];
	engyAlg1Errors[[i]] = Around[engyAlg1[[i]], stdAlg1[[i]]];
]
factorAlg2= Round[Abs[engyAlg2 - meanTheory] / stdTheory];
stdAlg2 = stdTheory;
engyAlg2Errors = engyAlg2;
engyTheoryErrors = engyTheory;
For[i = 1, i <= Length[labels], i++,
	If[factorAlg2 [[i]]> 1, stdAlg2[[i]]= factorAlg2[[i]]* stdTheory[[i]]];
	engyAlg2Errors[[i]] = Around[engyAlg2[[i]], stdAlg2[[i]]];
	engyTheoryErrors[[i]] = Around[engyTheory[[i]], stdTheory[[i]]]
]
colors = ColorData[54];
baseStyle = Directive[Black, FontFamily->"Arial", FontSize->16];
frameStyle = Append[baseStyle, Thickness@0.003];
rotateLabel[label_]:=Style[Rotate[label,Pi/4],30,Bold,Opacity[0.2],FontFamily->"Helvetica"]
BarChart[{engyMcErrors, engyAlg1Errors, engyAlg2Errors, engyTheoryErrors}, 
ChartLabels->{Placed[{"Mc","Alg1", "Alg2", "Theory"},Center, rotateLabel],labels},
BaseStyle->baseStyle, FrameStyle->frameStyle, ImageSize-> Large, ChartLegends->labels]


(* ::Input:: *)
(**)


PearsonChiSquareTest[engyMc, engyTheory]
PearsonChiSquareTest[engyAlg1, engyTheory]
PearsonChiSquareTest[engyAlg2, engyTheory]


PearsonChiSquareTest[Import["mc-lT.json", "RawJSON"]["energy_sample"], Import["a1-lT.json", "RawJSON"]["energy_sample"]]
PearsonChiSquareTest[Import["mc-lT.json", "RawJSON"]["energy_sample"], Import["a2-lT.json", "RawJSON"]["energy_sample"]]


magnMc = Import["mc-lT.json", "RawJSON"]["magnetization_sample"];
magnAlg1 = Import["a1-lT.json", "RawJSON"]["magnetization_sample"];
magnAlg2 = Import["a2-lT.json", "RawJSON"]["magnetization_sample"];
acrlMc = ListPlot[CorrelationFunction[magnMc, {20}], Filling -> Axis, PlotRange -> All, PlotLabel->"Mc"];
acrlAlg1 = ListPlot[CorrelationFunction[magnAlg1, {20}], Filling -> Axis, PlotRange -> All, PlotStyle->Magenta, PlotLabel->"Alg1"];
acrlAlg2 = ListPlot[CorrelationFunction[magnAlg2, {20}], Filling -> Axis, PlotRange -> All, PlotStyle->Brown, PlotLabel->"Alg2"];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


labels = Import["theory-lT.json", "RawJSON"]["magnetization_level"];
magnMc = Sort[Tally[Import["mc-lT.json", "RawJSON"]["magnetization_sample"]]][[All, 2]];
magnAlg1 = Sort[Tally[Import["a1-lT.json", "RawJSON"]["magnetization_sample"]]][[All, 2]];
magnAlg2 = Sort[Tally[Import["a2-lT.json", "RawJSON"]["magnetization_sample"]]][[All, 2]];
magnTheory = Round[Total[engyMc] * Import["theory-lT.json", "RawJSON"]["magnetization_probs"]];
stdTheory = StandardDeviation[MultinomialDistribution[n=Total[magnMc],Import["theory-lT.json", "RawJSON"]["magnetization_probs"]]];
meanTheory =  Mean[MultinomialDistribution[n= Total[engyMc],Import["theory-lT.json", "RawJSON"]["magnetization_probs"]]];
factorMc = Round[Abs[magnMc - meanTheory] / stdTheory];
stdMc = stdTheory;
magnMcErrors = magnMc;
For[i = 1, i <= Length[labels], i++,
	If[factorMc [[i]]> 1, stdMc[[i]] = factorMc[[i]]* stdTheory[[i]]];
	magnMcErrors[[i]] = Around[magnMc[[i]], stdMc[[i]]];
	]
factorAlg1 = Round[Abs[magnAlg1 - meanTheory] / stdTheory];
stdAlg1 = stdTheory;
magnAlg1Errors = magnAlg1;
For[i = 1, i <= Length[labels], i++, 
	If[factorAlg1[[i]]> 1, stdAlg1[[i]] = factorAlg1[[i]]* stdTheory[[i]]];
	magnAlg1Errors[[i]] = Around[magnAlg1[[i]], stdAlg1[[i]]];
]
factorAlg2= Round[Abs[magnAlg2 - meanTheory] / stdTheory];
stdAlg2 = stdTheory;
magnAlg2Errors = magnAlg2;
magnTheoryErrors = magnTheory;
For[i = 1, i <= Length[labels], i++,
	If[factorAlg2 [[i]]> 1, stdAlg2[[i]]= factorAlg2[[i]]* stdTheory[[i]]];
	magnAlg2Errors[[i]] = Around[magnAlg2[[i]], stdAlg2[[i]]];
	magnTheoryErrors[[i]] = Around[magnTheory[[i]], stdTheory[[i]]]
]
colors = ColorData[54];
baseStyle = Directive[Black, FontFamily->"Arial", FontSize->16];
frameStyle = Append[baseStyle, Thickness@0.003];
rotateLabel[label_]:=Style[Rotate[label,Pi/4],30,Bold,Opacity[0.2],FontFamily->"Helvetica"]
BarChart[{magnMcErrors, magnAlg1Errors, magnAlg2Errors, magnTheoryErrors}, 
ChartLabels->{Placed[{"Mc","Alg1", "Alg2", "Theory"},Center, rotateLabel],labels},
BaseStyle->baseStyle, FrameStyle->frameStyle, ImageSize-> Large, ChartLegends->labels]


PearsonChiSquareTest[magnMc, magnTheory]
PearsonChiSquareTest[magnAlg1, magnTheory]
PearsonChiSquareTest[magnAlg2, magnTheory]


PearsonChiSquareTest[Import["mc-lT.json", "RawJSON"]["magnetization_sample"], Import["a1-lT.json", "RawJSON"]["magnetization_sample"]]
PearsonChiSquareTest[Import["mc-lT.json", "RawJSON"]["magnetization_sample"], Import["a2-lT.json", "RawJSON"]["magnetization_sample"]]


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(*	*)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::Input:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)


(* ::InheritFromParent:: *)
(**)
