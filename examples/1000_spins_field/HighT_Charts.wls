#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[NotebookDirectory[]];
readSample[nameFile_, nameQuantity_] := Import[nameFile, "RawJSON"][nameQuantity];
acrPlot[sample_, length_, label_, color_] := ListPlot[CorrelationFunction[sample, {length}], Filling -> Axis, 
													  PlotRange -> All,  PlotStyle-> color, PlotLabel-> label];
kde[sample_, label_, color_] := Return[SmoothHistogram[sample, Automatic, "PDF", PlotStyle-> color, PlotLegends->{label}, PlotRange->All]];
hist[sample_, label_, color_, nameQuantity_] := Return[Histogram[sample, bspec = Automatic, ChartStyle->color, ChartLabels->{label}, 
															   AxesLabel->{nameQuantity, "Frequency"}]]


engyMc = readSample["mc-hT.json", "energy_sample"];
engyAlg1 = readSample["a1-hT.json", "energy_sample"];
engyAlg2 = readSample["a2-hT.json", "energy_sample"];
acrlMc = acrPlot[engyMc, 20, "Mc", Blue];
acrlAlg1 = acrPlot[engyAlg1, 20, "Alg1", Magenta];
acrlAlg2 = acrPlot[engyAlg2, 20, "Alg2", Brown];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


engyMcRescaled = engyMc - Min[engyMc];
engyAlg1Rescaled = engyAlg1 - Min[engyAlg1];
engyAlg2Rescaled = engyAlg2 - Min[engyAlg2];
mc = kde[engyMcRescaled, "Mc", Blue];
alg1 = kde[engyAlg1Rescaled, "Alg1", Magenta];
alg2 = kde[engyAlg1Rescaled, "Alg2", Brown];
Show[{mc, alg1, alg2}, AxesLabel->{"Energy level", "Density"}]


hMc = hist[engyMcRescaled, "Mc", LightBlue, "Energy level"];
hAlg1 = hist[engyAlg1Rescaled, "Alg1", LightMagenta, "Energy level"];
hAlg2 = hist[engyAlg1Rescaled, "Alg2", LightBrown, "Energy level"];
GraphicsRow[{hMc, hAlg1, hAlg2}, ImageSize->Full]


magnMc = readSample["mc-hT.json", "magnetization_sample"];
magnAlg1 = readSample["a1-hT.json", "magnetization_sample"];
magnAlg2 = readSample["a2-hT.json", "magnetization_sample"];
acrlMc = acrPlot[magnMc, 20, "Mc", Blue];
acrlAlg1 = acrPlot[magnAlg1, 20, "Alg1", Magenta];
acrlAlg2 = acrPlot[magnAlg2, 20, "Alg2", Brown];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


mc = kde[magnMc, "Mc", Blue];
alg1 = kde[magnAlg1, "Alg1", Magenta];
alg2 = kde[magnAlg2, "Alg2", Brown];
Show[{mc, alg1, alg2}, AxesLabel->{"Magnetization level", "Density"}]


hMc = hist[magnMc, "Mc", LightBlue, "Magnetization level"];
hAlg1 = hist[magnAlg1, "Alg1", LightMagenta, "Magnetization level"];
hAlg2 = hist[magnAlg2, "Alg2", LightBrown, "Magnetization level"];
GraphicsRow[{hMc, hAlg1, hAlg2}, ImageSize->Full]
