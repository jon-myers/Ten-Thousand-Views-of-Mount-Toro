


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
	logPitchEnv = Env.new(logPitch, envDurs, 'sine');
});

(
SynthDef.new(\moving_pluck, {|out=0|
	var spec, freqEnv, sig, transientEnv, allFreqs, pluckEnv, ampEnv;
	freqEnv = Env.newClear(40);
	pluckEnv = Env.newClear(40);
	ampEnv = Env.newClear(40);
	freqEnv = NamedControl.kr(\freqEnv, freqEnv.asArray);
	pluckEnv = NamedControl.kr(\pluckEnv, pluckEnv.asArray);
	ampEnv = NamedControl.kr(\ampEnv, ampEnv.asArray);
	freqEnv = 2 ** EnvGen.kr(freqEnv);
	sig = Pluck.ar(EnvGen.kr(ampEnv) * PinkNoise.ar(), EnvGen.kr(pluckEnv), 40.0.reciprocal, freqEnv.reciprocal, 5, 0.4, 1.0);
	// sig = sig * transientEnv;
	sig = Pan2.ar(sig);
	Out.ar(out, sig);
}).add;
)


Trig

~notes = 200*[1, 5/4, 4/3, 5/4, 4/3, 3/2, 4/3, 5/4, 1];
~pivots = [200*4/3, nil, 200*3/2, 200*3/2, 200*7/4, 200*7/4, 200*3/2, 200*4/3];
~durs = Array.fill(9, {0.35});
~waitProps = Array.fill(9, {1/2});
~pluckStarts = 0.35 * [0, 3, 5];
~pluckAmps = [0.7, 0.1, 0.4];



~env = ~freqEnv.value(~notes, ~pivots, ~durs, ~waitProps);
~pEnv = ~pluckEnv.value(~pluckStarts);
~aEnv = ~ampEnv.value(~pluckAmps, ~pluckStarts);
{EnvGen.kr(~pEnv)}.plot(3)
Synth(\moving_pluck, [\freqEnv, ~env, \pluckEnv, ~pEnv, \ampEnv, ~aEnv]);