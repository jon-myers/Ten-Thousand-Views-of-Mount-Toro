(~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';
);
(
~pluck = File.open(~jsonPath ++ "moving_pluck_phrase.JSON", "r");
~pluck = ~pluck.readAllString.parseYAML;
~notes = ~pluck["notes"].asFloat;
~pivots = ~pluck["pivots"];
~pivots = Array.fill(~pluck["pivots"].size, {|i|
	if(~pluck["pivots"][i] == "nil", {nil}, {~pluck["pivots"][i].asFloat})
});
~durs = ~pluck["durs"].asFloat;
~waitProps = ~pluck["waitProps"].asFloat;
~pluckStarts = ~pluck["pluckStarts"].asFloat;
~pluckAmps = ~pluck["pluckAmps"].asFloat;
)

~notes = 200*[1, 5/4, 4/3, 5/4, 4/3, 3/2, 4/3, 5/4, 1];
~pivots = [200*4/3, nil, nil, nil, 200*7/4, 200*7/4, 200*3/2, 200*4/3];
~durs = Array.fill(9, {0.35});
~waitProps = Array.fill(9, {1/2});
~pluckStarts = 0.35 * [0, 3, 5];
~pluckAmps = [0.7, 0.7, 0.7];

~notes = 100*[16/8, 15/8, 16/8];
~pivots = [nil, nil];
~durs = [0.5, 1, 0.5];
~waitProps = [0, 0.5];
~pluckStarts = [0];
~pluckAmps = [0.75];

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
})

(
~ampEnv = {|amps, starts|
	var durs;
	durs = Array.fill(starts.size-1, {|i| starts[i+1] - starts[i]});
	Env.new(amps, durs, 'hold');
})

// {EnvGen.kr(~ampEnv.value([1, 0.5, 0.75], [0, 0.25, 0.6]))}.plot(2)


// {EnvGen.kr(~pluckEnv.value([0, 1, 1.5]))}.plot(3)


(// staying in log the whole time now, so I only have to make one EngGen.
~freqEnv = {|notes, pivots, durs, waitProps|
	var envDurs = [], envSineDurs = [], envCenters = [], envWidths = [], envSines = [], envSineCurves = [];
	var centerEnv, widthEnv, sineEnv, logPitch, logPitchEnv, test;
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
	logPitch.postln;
	logPitchEnv = Env.new(logPitch, envDurs, 'sine');
});

(
SynthDef.new(\moving_pluck, {|out=0, releaseDur|
	var spec, freqEnv, sig, transientEnv, allFreqs, pluckEnv, ampEnv, doneEnv;
	freqEnv = Env.newClear(200);
	pluckEnv = Env.newClear(200);
	ampEnv = Env.newClear(200);
	doneEnv = EnvGen.kr(Env.new([0, 0], [releaseDur]), doneAction: 2);
	freqEnv = NamedControl.kr(\freqEnv, freqEnv.asArray);
	pluckEnv = NamedControl.kr(\pluckEnv, pluckEnv.asArray);
	ampEnv = NamedControl.kr(\ampEnv, ampEnv.asArray);
	freqEnv = 2 ** EnvGen.kr(freqEnv);
	sig = Pluck.ar(EnvGen.kr(ampEnv) * PinkNoise.ar(), EnvGen.kr(pluckEnv), 40.0.reciprocal, freqEnv.reciprocal, 5, 0.38, 1.0);
	// sig = Streson.ar(sig, (2**~env.asArray[0]).reciprocal);
	// sig = HPF.ar(sig, 300);
	// sig = sig * transientEnv;
	sig = Pan2.ar(sig);
	Out.ar(out, sig);
	Out.ar(~testSympatheticBus.index, sig);
}).add;
)

SynthDef.new(\busOut, {|out=0|
	var sig;
	sig = ~testSympatheticBus.ar(1) * [0, 1];
	sig = Streson.ar(sig, (2**~env.[0]).reciprocal);
	Out.ar(out, sig*[0, 1])
}).add

Synth(\busOut)
~testSympatheticBus = Bus.audio(s, 1);


(
~pluck = File.open(~jsonPath ++ "moving_pluck_phrase.JSON", "r");
~pluck = ~pluck.readAllString.parseYAML;
~notes = ~pluck["notes"].asFloat;
~pivots = ~pluck["pivots"];
~pivots = Array.fill(~pluck["pivots"].size, {|i|
	if(~pluck["pivots"][i] == "nil", {nil}, {~pluck["pivots"][i].asFloat})
});
~releaseDur = ~pluck["releaseDur"].asFloat;
~durs = ~pluck["durs"].asFloat;
~waitProps = ~pluck["waitProps"].asFloat;
~pluckStarts = ~pluck["pluckStarts"].asFloat;
~pluckAmps = ~pluck["pluckAmps"].asFloat;

~env = ~freqEnv.value(~notes, ~pivots, ~durs, ~waitProps);
~pEnv = ~pluckEnv.value(~pluckStarts);
~aEnv = ~ampEnv.value(~pluckAmps, ~pluckStarts);
);

{EnvGen.kr(~env)}.plot(3)
~releaseDur = 3.0
Synth(\moving_pluck, [\releaseDur, ~releaseDur, \freqEnv, ~env, \pluckEnv, ~pEnv, \ampEnv, ~aEnv]);