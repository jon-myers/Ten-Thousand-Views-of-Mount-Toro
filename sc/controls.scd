(
(
p = Pbind(
	\instrument, \singlePop,
	\attack, Pseq(~pc_attacks, 1),
	\decay, Pseq(~pc_decays, 1),
	\lpFilterFreq, Pseq(~pc_lpfs, 1),
	\hpFilterFreq, Pseq(~pc_hpfs, 1),
	\dur, Pseq(~pc_durs, 1)
).play
);

(
q = Pbind(
	\freqs, Pseq(~k_freqs, 1),
	\decays, Pseq(~k_decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~k_durs, 1),
	\off_dur, Pseq(8*(~k_next_durs + ~k_durs), 1),
	\pan,Pseq(~k_pan, 1),
	\amps, Pseq(~k_amps*0.5, 1),
	\transient_dur, Prand([0.005], inf),
	\transient_curve, Prand([-4, -3, -2, -1, 0, 1, 2, 3, 4], inf)
).play
);

(
r = Pbind(
	\instrument, \moving_pluck,
	\freqEnv, Pseq(~allFreqEnvs, 1),
	\pluckEnv, Pseq(~allPluckEnvs, 1),
	\ampEnv, Pseq(~allAmpEnvs, 1),
	\releaseDur, Pseq(~allReleaseDurs, 1),
	\dur, Pseq(~dursBetweenOnsets, 1)
).play
)

)