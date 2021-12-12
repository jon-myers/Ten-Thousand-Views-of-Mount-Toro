

(var server, p_def, mp_def, k_defs;
(
(// klank
// ~dir = PathName.new(Document.current.dir);
~dir = PathName("/root/azure/sc"); // this needs to change depending on machine
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~klanks = File.open(~jsonPath ++ "klank_packets_alt.JSON", "r");
~klanks = ~klanks.readAllString.parseYAML;
~k_freqs = Array.fill(~klanks.size, {|i| [~klanks[i]["freqs"].asFloat]});
~k_amps = Array.fill(~klanks.size, {|i| [~klanks[i]["amps"].asFloat]});
~k_pan = Array.fill(~klanks.size, {|i| ~klanks[i]["pan"].asFloat});

~k_durs = Array.fill(~klanks.size, {|i|
	var dur = ~klanks[i]["rt_dur"];
	dur = if(dur[0].asString=="R", {dur.interpret}, {dur.asFloat});
	dur});

~k_next_durs = Array.fill(~klanks.size - 1, {|i|
	var dur = ~klanks[i+1]["rt_dur"];
	dur = if(dur[0].asString=="R", {dur.interpret}, {dur.asFloat})});
~k_next_durs = ~k_next_durs.add( ~k_durs[~k_durs.size - 1]);

~k_decays = Array.fill(~klanks.size, {|i| [~klanks[i]["rt_decays"]].asFloat});
~k_transient_durs = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_dur"].asFloat});
~k_transient_curve = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_curve"].asFloat});
);

(// popcorn
~kernals = File.open(~jsonPath ++ "kernals.JSON", "r");
~kernals = ~kernals.readAllString.parseYAML;
~pc_attacks = Array.fill(~kernals.size, {|i| ~kernals[i]["attack"].asFloat});
~pc_hpfs = Array.fill(~kernals.size, {|i| ~kernals[i]["hp_freq"].asFloat});
~pc_lpfs = Array.fill(~kernals.size, {|i| ~kernals[i]["lp_freq"].asFloat});
~pc_decays = Array.fill(~kernals.size, {|i| i.postln; ~kernals[i]["rt_dur"].asFloat - ~pc_attacks[i]});
~pc_durs = Array.fill(~kernals.size-1, {|i| ~kernals[i+1]["rt_onset_time"].asFloat - ~kernals[i]["rt_onset_time"].asFloat});
~pc_durs.add(~pc_durs[~pc_durs.size-1]);
~pc_durs = Array.fill(~pc_durs.size, {|i|
	if (~kernals[i]["type"] == "rest",
		{("Rest(" ++ ~pc_durs[i].asString ++ ")").interpret},
		{~pc_durs[i]})
});
~pc_amps = Array.fill(~kernals.size, {|i| ~kernals[i]["amp"].asFloat});
~pc_pans = Array.fill(~kernals.size, {|i| ~kernals[i]["pan"].asFloat});
);

(// moving pluck helper: these three functions return envs that get passed to the synthdef via Pbind.
(
~pluckEnv = {|starts|
	var durs, levels, envDurs, pluckEnv;
	durs = Array.fill(starts.size - 1, {|i| starts[i+1] - starts[i]});
	levels = [];
	envDurs = [];
	starts.size.do({|i|
		levels = levels.insert(levels.size, -1);
		levels = levels.insert(levels.size, 1);
		if(i < (starts.size - 1), {
			envDurs = envDurs.insert(envDurs.size, 0);
			envDurs = envDurs.insert(envDurs.size, durs[i]);
		}, {
			envDurs = envDurs.insert(envDurs.size, 0);
		});
	});
	pluckEnv = Env.new(levels, envDurs);
	pluckEnv
});

(
~ampEnv = {|amps, starts|
	var durs;
	durs = Array.fill(starts.size-1, {|i| starts[i+1] - starts[i]});
	Env.new(amps, durs, 'hold');
});

(
~freqEnv = {|notes, pivots, durs, waitProps|
	var envDurs = [], envSineDurs = [], envCenters = [], envWidths = [], envSines = [], envSineCurves = [];
	var centerEnv, widthEnv, sineEnv, logPitch, logPitchEnv, test;
	if(notes.size == 1) {
		logPitchEnv = Env.new([notes[0].log2, notes[0].log2], [durs[0]])
	} {
		(notes.size-1).do({|i|
			var wait, move, test;
			wait = durs[i] * waitProps[i];
			move = durs[i] * (1 - waitProps[i]);

			if(pivots[i] == nil, {var logAvg;
				envDurs = envDurs ++ [wait, move];
				envSineDurs = envSineDurs ++ [wait, 0, move, 0];
				logAvg = (notes[i].log2 + notes[i+1].log2) / 2;
				envCenters = envCenters ++ [notes[i].log2, logAvg];
				envWidths = envWidths ++ [0, (notes[i].log2 - logAvg).abs];
				if(i==0,
					{envSines = envSines ++ if(notes[i].log2 > logAvg, {[0, 1, -1]}, {[0, -1, 1]})},
					{envSines = envSines ++ if(notes[i].log2 > logAvg, {[1, -1]}, {[-1, 1]})});
				envSineCurves = envSineCurves ++ ['hold', 'hold', 'sine', 'hold'];
			}, {var logAvg;
				envDurs = envDurs ++ [wait, move/2, move/2];
				envSineDurs = envSineDurs ++ [wait, 0, move/2, move/2, 0];
				logAvg = [(notes[i].log2 + pivots[i].log2) / 2, (pivots[i].log2 + notes[i+1].log2) / 2];
				envCenters = envCenters ++ [notes[i].log2, logAvg[0], logAvg[1]];
				envWidths = envWidths ++ [0, (notes[i].log2 - logAvg[0]).abs, (notes[i+1].log2 - logAvg[1]).abs];
				if(i==0,
					{envSines = envSines ++ [0, -1, 1, -1]},
					{envSines = envSines ++ [-1, 1, -1]});
				envSineCurves = envSineCurves ++ ['hold', 'hold', 'sine', 'sine', 'hold'];
			})
		});

		envCenters.insert(0, envCenters[0]);
		envWidths.insert(0, envWidths[0]);
		logPitch = Array.fill(envCenters.size, {|i| envCenters[i] + (envWidths[i] * envSines[i])});
		logPitchEnv = Env.new(logPitch, envDurs, 'sine');
	};
	logPitchEnv
});
);

(// moving pluck
(
~phrases = File.open(~jsonPath ++ "moving_pluck_phrases.JSON", "r");
~phrases = ~phrases.readAllString.parseYAML;
~dursBetweenOnsets = Array.fill(~phrases.size - 1, {arg i; ~phrases[i+1]["rt_start"].asFloat - ~phrases[i]["rt_start"].asFloat});
~dursBetweenOnsets = ~dursBetweenOnsets.add(~phrases[~phrases.size-1]["rt_durTot"].asFloat);
~floatifyPivots = {|pivots|
	var outPivots;
	outPivots = Array.fill(pivots.size, {|i| if(pivots[i] == "nil", {nil}, {pivots[i].asFloat})});
	outPivots
};
~allNotes = Array.fill(~phrases.size, {|i| ~phrases[i]["notes"].asFloat});
~allPivots = Array.fill(~phrases.size, {|i| ~floatifyPivots.value(~phrases[i]["pivots"])});
~allDurs = Array.fill(~phrases.size, {|i| ~phrases[i]["rt_durs"].asFloat});
~allWaitProps = Array.fill(~phrases.size, {|i| ~phrases[i]["waitProps"].asFloat});
~allPluckStarts = Array.fill(~phrases.size, {|i| ~phrases[i]["rt_pluckStarts"].asFloat});
~allPluckAmps = Array.fill(~phrases.size, {|i| ~phrases[i]["pluckAmps"].asFloat});

~allFreqEnvs = Array.fill(~phrases.size, {|i|
	[~freqEnv.value(~allNotes[i], ~allPivots[i], ~allDurs[i], ~allWaitProps[i])]
});
~allPluckEnvs = Array.fill(~phrases.size, {|i| [~pluckEnv.value(~allPluckStarts[i])]});
~allAmpEnvs = Array.fill(~phrases.size, {|i| [~ampEnv.value(~allPluckAmps[i], ~allPluckStarts[i])]});
~allReleaseDurs = Array.fill(~phrases.size, {|i| ~phrases[i]["rt_releaseDur"].asFloat});
~sympFreqs = Array.fill(~phrases.size, {|i| [~phrases[i]["symps"].asFloat]});
~coefs = Array.fill(~phrases.size, {|i| ~phrases[i]["coef"].asFloat});
)
);

server = Server(\nrt,
	options: ServerOptions.new.numOutputBusChannels_(2).numInputBusChannels_(2)
);
k_defs = Array.fill(7, {|n|
	var name = \klank_ ++ (n+1);
	SynthDef(name, {|out=0, off_dur=2.0, pan=0.0, transient_dur=0.001, transient_curve=0|
		var freqs, decays, amps, sig, sig2, timer, test_dur=2.0, input, input_env;
		// pan.poll;
		freqs = NamedControl.kr(\freqs, 440.0!(n+1));
		decays = NamedControl.kr(\decays, 0.5!(n+1));
		amps = NamedControl.kr(\amps, 0.5!(n+1));
		timer = EnvGen.kr(Env([0,1, 1], [transient_dur, off_dur - transient_dur], transient_curve), doneAction:2);
		input_env = EnvGen.kr(Env([0, 0.01, 0], [transient_dur, off_dur - transient_dur]));
		input = Impulse.ar(0, 0, 0.2);
		sig = Klank.ar(`[freqs, amps, decays], input);
		sig = Pan2.ar((sig) * timer, pan);
		Out.ar(out, sig);
	})
});
p_def = SynthDef(\singlePop, {|out=0, attack=0.01, decay=1.0, curve = -4, lpFilterFreq=1600, hpFilterFreq=50, amp=0.5, pan=0.0|
	var sig, env, envGen;
	env = Env.perc(attack, decay);
	envGen = EnvGen.kr(env, 1.0, doneAction:2);
	sig = PinkNoise.ar();
	sig = LPF.ar(sig, lpFilterFreq);
	sig = HPF.ar(sig, hpFilterFreq);
	sig = envGen * sig;
	sig = 2 * sig * amp;
	sig = Pan2.ar(sig, pan);
	Out.ar(out, sig);
});
mp_def = SynthDef(\moving_pluck, {|out=0, releaseDur, coef|
	var spec, freqEnv, sig, transientEnv, allFreqs, pluckEnv, ampEnv, doneEnv, symps, sympFreqs, sympPans;
	freqEnv = Env.newClear(200);
	pluckEnv = Env.newClear(200);
	ampEnv = Env.newClear(200);
	doneEnv = EnvGen.kr(Env.new([0, 0], [releaseDur]), doneAction: 2);
	freqEnv = NamedControl.kr(\freqEnv, freqEnv.asArray);
	pluckEnv = NamedControl.kr(\pluckEnv, pluckEnv.asArray);
	ampEnv = NamedControl.kr(\ampEnv, ampEnv.asArray);
	sympFreqs = NamedControl.kr(\sympFreqs, 200.0!21);
	sympPans = Array.fill(21, {2.0.rand-1});
	freqEnv = 2 ** EnvGen.kr(freqEnv);
	transientEnv = EnvGen.kr(Env.new([0, 1, 1], [0.2, 10]), EnvGen.kr(pluckEnv));
	sig = 0.5 * Pluck.ar(EnvGen.kr(ampEnv) * PinkNoise.ar(), EnvGen.kr(pluckEnv), 40.0.reciprocal, freqEnv.reciprocal, 5, coef, 1.0);
	sig = BHiPass.ar(sig, 400);
	sig = BLowPass.ar(sig, 3000);
	symps = Mix.ar(Array.fill(21, {|i|
		var symp;
		symp = AllpassL.ar(sig, sympFreqs[i].reciprocal, sympFreqs[i].reciprocal, 10, 0.4);
		symp = Pan2.ar(symp, sympPans[i]);
	}));
	sig = Pan2.ar(sig);
	sig = Mix.ar([sig, symps]);
	Out.ar(out, sig);
});
k_defs.do{|def| def.add};
p_def.add;
mp_def.add;

q = Pbind(
	\freqs, Pseq(~k_freqs, 1),
	\decays, Pseq(~k_decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~k_durs, 1),
	\off_dur, Pseq(16*(~k_next_durs + ~k_durs), 1),
	\pan,Pseq(~k_pan, 1),
	\amps, Pseq(~k_amps, 1),
	\transient_dur, Prand([0.005], inf),
	\transient_curve, Prand([-4, -3, -2, -1, 0, 1, 2, 3, 4], inf)
);

p = Pbind(
	\instrument, \singlePop,
	\attack, Pseq(~pc_attacks, 1),
	\decay, Pseq(~pc_decays, 1),
	\lpFilterFreq, Pseq(~pc_lpfs, 1),
	\hpFilterFreq, Pseq(~pc_hpfs, 1),
	\dur, Pseq(~pc_durs, 1),
	\amp, Pseq(~pc_amps, 1),
	\pan, Pseq(~pc_pans, 1)
);

r = Pbind(
	\instrument, \moving_pluck,
	\freqEnv, Pseq(~allFreqEnvs, 1),
	\pluckEnv, Pseq(~allPluckEnvs, 1),
	\ampEnv, Pseq(~allAmpEnvs, 1),
	\releaseDur, Pseq(~allReleaseDurs, 1),
	\dur, Pseq(~dursBetweenOnsets, 1),
	\sympFreqs, Pseq(~sympFreqs, 1),
	\coef, Pseq(~coefs, 1)
);

a = Ppar([q, p, r]).asScore(thisProcess.argv[0].asFloat, timeOffset:0.001);

k_defs.do{|def| a.add([0.0, [\d_recv, def.asBytes]])};
a.add([0.0, [\d_recv, p_def.asBytes]]);
a.add([0.0, [\d_recv, mp_def.asBytes]]);
a.sort;
a.recordNRT(
	outputFilePath: ("/root/audioGeneration/" ++ thisProcess.argv[1] ++ ".wav").standardizePath,
	sampleRate: 48000,
	headerFormat: "WAV",
	sampleFormat: "int16",
	options: server.options,
	// duration: 40 * 60
	action: {0.exit}
);
server.remove;
)
)
