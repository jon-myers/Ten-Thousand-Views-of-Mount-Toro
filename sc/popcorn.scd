(
~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~kernals = File.open(~jsonPath ++ "kernals.JSON", "r");
~kernals = ~kernals.readAllString.parseYAML;

~pc_attacks = Array.fill(~kernals.size, {|i| ~kernals[i]["attack"].asFloat});
~pc_hpfs = Array.fill(~kernals.size, {|i| ~kernals[i]["hp_freq"].asFloat});
~pc_lpfs = Array.fill(~kernals.size, {|i| ~kernals[i]["lp_freq"].asFloat});
~pc_decays = Array.fill(~kernals.size, {|i| ~kernals[i]["rt_dur"].asFloat - ~pc_attacks[i]});
~pc_durs = Array.fill(~kernals.size-1, {|i| ~kernals[i+1]["rt_onset_time"].asFloat - ~kernals[i]["rt_onset_time"].asFloat});
~pc_durs.add(~pc_durs[~pc_durs.size-1])
)

~kernals[0].keys()
~lpfs[0]
~hpfs[0]

/*
(
SynthDef.new(\popcorn, {|out=0, density=1|
	var sig, env, envGen, rands, trig, randVal;
	rands = Array.fill(1000, {1.0.rand});
	trig = Dust.kr(density);
	randVal = Select.kr(Stepper.kr(trig, 0, 0, 1000), rands).poll;
	env = Env.perc(0.01, 0.125 * (2 ** (4 * randVal)));
	envGen = EnvGen.kr(env, trig);
	sig = envGen * PinkNoise.ar(0.2);
	Out.ar(out, sig);
}).add;
)

x = Synth(\popcorn, [\density, 2]);

{ Stepper.kr(Impulse.kr(100), 0,1,10,1) }.plot(0.3,minval:0,maxval:10);*/


(
SynthDef.new(\singlePop, {|out=0, attack=0.01, decay=1.0, curve = -4, lpFilterFreq=1600, hpFilterFreq=50|
	var sig, env, envGen;
	env = Env.perc(attack, decay);
	envGen = EnvGen.kr(env, 1.0, doneAction:2);
	sig = PinkNoise.ar();
	sig = LPF.ar(sig, lpFilterFreq);
	sig = HPF.ar(sig, hpFilterFreq);
	sig = envGen * sig;
	sig = Pan2.ar(sig);
	Out.ar(out, sig);
}).add
)

Synth(\singlePop, [\attack, 0.16, \decay, 0.25, \curve, -4, \lpFilterFreq, 80 * (2 ** 9), \hpFilterFreq, 1000])

(
p = Pbind(
	\instrument, \singlePop,
	\attack, Pseq(~pc_attacks, 1),
	\decay, Pseq(~pc_decays, 1),
	\lpFilterFreq, Pseq(~pc_lpfs, 1),
	\hpFilterFreq, Pseq(~pc_hpfs, 1),
	\dur, Pseq(~pc_durs, 1)
).play
)

80 * (2 ** 9)