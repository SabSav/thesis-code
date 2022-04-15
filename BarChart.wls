#!/usr/bin/env wolframscript
(* ::Package:: *)

Quiet@Remove@"`*";
Row@{"File: ",
  files = FileNameJoin[{
    NotebookDirectory[], "data", # <> ".json"
  }]& /@ {"mc"}
}

tallies = Tally[
  Import[#, "RawJSON"]["energy_sample"]
] & /@ files;

(* These are just two stubs for missing data *)

AppendTo[tallies, First@tallies];
AppendTo[tallies, First@tallies];

(* Add error bars to bars*)
withErrors = Function[tally, Around[#[[2]], 1000] & /@ tally] /@ tallies;
bars = MapThread[{#1, #2, #3} &, withErrors];

colors = ColorData[54];
baseStyle = Directive[Black, FontFamily->"Arial", FontSize->16];
frameStyle = Append[baseStyle, Thickness@0.003];

filledSquare=Graphics[{EdgeForm[None], Rectangle[]}, PlotRangePadding->0];
filledDiamond=Graphics[Rotate[Rectangle[], 45*Degree], PlotRangePadding->0];
filledCircle=Graphics[{EdgeForm[None], Disk[]}, PlotRangePadding->0];
filledTriangle=Graphics[Triangle@{{0,0}, {1,0}, {0.5,1}}, PlotRangePadding->0];
emptyCircle=Graphics[{Circle}, PlotRangePadding->Scaled[0.1]];
emptySquare=Graphics[{
	AbsoluteThickness[2], Line@{{0,0}, {1,0}, {1,1}, {0,1}, {0,0}}
}, PlotRangePadding->0];
emptyDiamond=Graphics[{
	AbsoluteThickness[2], Rotate[Line@{{0,0}, {1,0}, {1,1}, {0,1}, {0,0}}, 45*Degree]
}, PlotRangePadding->0];
cross=Graphics[{
	Thickness[0.1], Line@{{0,0}, {1,1}}, Line@{{1,0}, {0,1}}
}, PlotRangePadding->0];

solid[clr_] := Directive[clr, Dashing[{}], Thickness@0.005]
dashed[clr_] := Directive[clr, Dashing[{0.04, 0.02}], Thickness@0.008]

BarChart[bars,
	BaseStyle->baseStyle, FrameStyle->frameStyle,
	ImageSize->Large, Frame->True
]



