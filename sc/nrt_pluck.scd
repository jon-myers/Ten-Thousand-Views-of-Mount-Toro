(// these three functions return envs that get passed to the synthdef via Pbind.
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

(// staying in log the whole time now, so I only have to make one EngGen.
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
~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';



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

(
var server = Server(\nrt,
	options: ServerOptions.new.numOutputBusChannels_(2).numInputBusChannels_(2)
),
def = SynthDef(\moving_pluck, {|out=0, releaseDur, coef|
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
def.add;

r = Pbind(
	\instrument, \moving_pluck,
	\freqEnv, Pseq(~allFreqEnvs, 1),
	\pluckEnv, Pseq(~allPluckEnvs, 1),
	\ampEnv, Pseq(~allAmpEnvs, 1),
	\releaseDur, Pseq(~allReleaseDurs, 1),
	\dur, Pseq(~dursBetweenOnsets, 1),
	\sympFreqs, Pseq(~sympFreqs, 1),
	\coef, Pseq(~coefs, 1)
).asScore(40 * 60, timeOffset:0.001);

r.add([0.0, [\d_recv, def.asBytes]]);
r.sort;

r.recordNRT(
	outputFilePath: "~/nrt_plucks.wav".standardizePath,
	sampleRate: 48000,
	headerFormat: "WAV",
	sampleFormat: "int16",
	options: server.options,
	duration: 40 * 60
);
server.remove;
)










