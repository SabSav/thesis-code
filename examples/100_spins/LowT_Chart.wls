#!/usr/bin/env wolframscript
(* ::Package:: *)

SetDirectory[NotebookDirectory[]];
engyMc = Import["mc-lT.json", "RawJSON"]["energy_sample"];
engyAlg1 = Import["a1-lT.json", "RawJSON"]["energy_sample"];
engyAlg2 = Import["a2-lT.json", "RawJSON"]["energy_sample"];
acrlMc = ListPlot[CorrelationFunction[engyMc, {500}], Filling -> Axis, PlotRange -> All, PlotLabel->"Mc"];
acrlAlg1 = ListPlot[CorrelationFunction[engyAlg1, {500}], Filling -> Axis, PlotRange -> All, PlotStyle->Magenta, PlotLabel->"Alg1"];
acrlAlg2 = ListPlot[CorrelationFunction[engyAlg2, {500}], Filling -> Axis, PlotRange -> All, PlotStyle->Brown, PlotLabel->"Alg2"];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


SetDirectory[NotebookDirectory[]];
engyMc = Import["mc-lT.json", "RawJSON"]["energy_sample"];
engyAlg1 = Import["a1-lT.json", "RawJSON"]["energy_sample"];
engyAlg2 = Import["a2-lT.json", "RawJSON"]["energy_sample"];

mc = SmoothHistogram[engyMc, Automatic, "PDF", PlotLegends->{"MC"}];
alg1 = SmoothHistogram[engyAlg1, Automatic, "PDF", PlotStyle->Magenta, PlotLegends->{"Alg1"}];
alg2 = SmoothHistogram[engyAlg2, Automatic,"PDF", PlotStyle-> Brown, PlotLegends->{"Alg2"}];
Show[mc, alg1, alg2]


J = Import["mc-lT.json", "RawJSON"]["coupling"];
engyMcRescaled = engyMc - Min[engyMc];
engyAlg1Rescaled = engyAlg1 - Min[engyAlg1];
engyAlg2Rescaled = engyAlg2 - Min[engyAlg2];
hMc = Histogram[engyMcRescaled, bspec = 4, ChartStyle->LightBlue, ChartLabels->{"Mc"}];
hAlg1 = Histogram[engyAlg1Rescaled, bspec = 4, ChartStyle->LightMagenta, ChartLabels->{"Alg1"}];
hAlg2 = Histogram[engyAlg2Rescaled, bspec = 4, ChartStyle->LightBrown, ChartLabels->{"Alg2"}];
GraphicsRow[{hMc, hAlg1, hAlg2}]


(*No reasons to pass the counts, the Pearson  test effectively compares a histogram of data to a theoretical histogram 
based on dist. The bins are chosen to have equal probability in dist. \[RightGuillemet]*)
PearsonChiSquareTest[engyMc, engyAlg1]
PearsonChiSquareTest[engyMc, engyAlg2]


magnMc = Import["mc-lT.json", "RawJSON"]["magnetization_sample"];
magnAlg1 = Import["a1-lT.json", "RawJSON"]["magnetization_sample"];
magnAlg2 = Import["a2-lT.json", "RawJSON"]["magnetization_sample"];
acrlMc = ListPlot[CorrelationFunction[magnMc, {50}], Filling -> Axis, PlotRange -> All, PlotLabel->"Mc"];
acrlAlg1 = ListPlot[CorrelationFunction[magnAlg1, {1000}], Filling -> Axis, PlotRange -> All, PlotStyle->Magenta, PlotLabel->"Alg1"];
acrlAlg2 = ListPlot[CorrelationFunction[magnAlg2, {50}], Filling -> Axis, PlotRange -> All, PlotStyle->Brown, PlotLabel->"Alg2"];
GraphicsRow[{acrlMc, acrlAlg1, acrlAlg2}, ImageSize->Full]


SetDirectory[NotebookDirectory[]];
magnMc = Import["mc-lT.json", "RawJSON"]["magnetization_sample"];
magnAlg1 = Import["a1-lT.json", "RawJSON"]["magnetization_sample"];
magnAlg2 = Import["a2-lT.json", "RawJSON"]["magnetization_sample"];

mc = SmoothHistogram[magnMc, Automatic, "PDF", PlotLegends->{"MC"}];
alg1 = SmoothHistogram[magnAlg1, Automatic, "PDF", PlotStyle->Magenta, PlotLegends->{"Alg1"}];
alg2 = SmoothHistogram[magnAlg2, Automatic,"PDF", PlotStyle-> Brown, PlotLegends->{"Alg2"}];
Show[mc, alg1, alg2]


Nsamples = Length[Import["mc-lT.json", "RawJSON"]["energy_sample"]];
magnMcRescaled = magnMc - Min[magnMc];
magnAlg1Rescaled = magnAlg1 - Min[magnAlg1];
magnAlg2Rescaled = magnAlg2 - Min[magnAlg2];
 
hMc = Histogram[magnMcRescaled, ChartStyle->LightBlue, ChartLabels->{"Mc"}];
hAlg1 = Histogram[magnAlg1Rescaled, ChartStyle->LightMagenta, ChartLabels->{"Alg1"}];
hAlg2 = Histogram[magnAlg2Rescaled, ChartStyle->LightBrown, ChartLabels->{"Alg2"}];
GraphicsRow[{hMc, hAlg1, hAlg2}]


(*No reasons to pass the counts, the Pearson  test effectively compares a histogram of data to a theoretical histogram 
based on dist. The bins are chosen to have equal probability in dist. \[RightGuillemet]*)
PearsonChiSquareTest[magnMc, magnAlg1]
PearsonChiSquareTest[magnMc, magnAlg2]






