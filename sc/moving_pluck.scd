
(
~log_avg = {|pt_a, pt_b|
	pt_a * (2 ** ((pt_b/pt_a).log2 / 2))
});

(
~freqEnv = {|notes, pivots, durs, waitProps|
	var envDurs = [], envSineDurs = [], envCenters = [], envWidths = [], envSines = [], envSineCurves = [];
	var centerEnv, widthEnv, sineEnv;
	(notes.size-1).do({|i|
		var wait, move;
		wait = durs[i] * waitProps[i];
		move = durs[i] * (1 - waitProps[i]);
		if(pivots[i] == nil, {var logAvg;
			envDurs = envDurs ++ [wait, move];
			envSineDurs = envSineDurs ++ [wait, move, 0];
			logAvg = ~log_avg.value(notes[i], notes[i+1]);
			envCenters = envCenters ++ [notes[i], logAvg];
			envWidths = envWidths ++ [0, (notes[i]/logAvg).log2.abs];
			envSines = envSines ++ if(notes[i] > logAvg, {[0, 1, -1]}, {[0, -1, 1]});
			envSineCurves = envSineCurves ++ ['hold', 'sine', 'sine'];
		}, {var logAvg;
			envDurs = envDurs ++ [wait, move/2, move/2];
			envSineDurs = envSineDurs ++ [wait, move/2, move/2, 0];
			logAvg = [~log_avg.value(notes[i], pivots[i]), ~log_avg.value(pivots[i], notes[i+1])];
			envCenters = envCenters ++ [notes[i], logAvg[0], logAvg[1]];
			envWidths = envWidths ++ [0, (notes[i]/logAvg[0]).log2.abs, (notes[i+1]/logAvg[1]).log2.abs];
			envSines = envSines ++ [0, -1, 1, -1];
			envSineCurves = envSineCurves ++ ['hold', 'sine', 'sine', 'sine'];
		})
	});
	envDurs.postln;
	envCenters = envCenters.add(notes[notes.size-1]);
	envWidths = envWidths.add(0);
	envSineDurs = envSineDurs[..envSineDurs.size-2];
	envSineCurves = envSineCurves[..envSineCurves.size-2];
	centerEnv = EnvGen.kr(Env.new(envCenters, envDurs, 'hold'));
	widthEnv = EnvGen.kr(Env.new(envWidths, envDurs, 'hold'));
	sineEnv = EnvGen.kr(Env.new(envSines, envSineDurs, envSineCurves));
	centerEnv * (2.0 ** (sineEnv * widthEnv));
});



~notes = 200*[1, 5/4, 4/3, 5/4, 4/3, 3/2, 4/3, 5/4, 1];
~pivots = [200*4/3, 200*3/2, 200*3/2, 200*3/2, 200*7/4, 200*7/4, 200*3/2, 200*4/3];
~durs = Array.fill(9, {0.35});
~waitProps = Array.fill(9, {1/3});


(
SynthDef.new(\moving_pluck, {|out=0|
	var spec, freqEnv, sig, transientEnv, allFreqs, notes, pivots, durs, waitProps;
	notes = NamedControl.kr(\notes, ~notes);
	pivots = NamedControl.kr(\pivots, ~pivots);
	durs = NamedControl.kr(\durs, ~durs);
	waitProps = NamedControl.kr(\waitProps, ~waitProps);
	freqEnv = ~freqEnv.value(notes, pivots, durs, waitProps);
	// transientEnv = EnvGen.kr(Env.new([0, 1], [0.05]));
	sig = Pluck.ar(WhiteNoise.ar(0.5), Impulse.kr(0), 50.0.reciprocal, freqEnv.reciprocal, 20, 0.6, 1.0);
	// sig = sig * transientEnv;
	Out.ar(out, sig);
}).add;
)




Synth(\moving_pluck, [\notes, ~notes, \pivots, ~pivots, \durs, ~durs, \waitProps, ~waitProps]);






~freqEnv.value(~sample_spec['notes'], ~sample_spec['pivots'], ~sample_spec['durs'], ~sample_spec['waitProps']);
~sample_spec['notes']

{~freqEnv.value(~notes, ~pivots, ~durs, ~waitProps)}.plot(3)
[1, 2, 3].minItem

