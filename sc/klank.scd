(
~dels = Array.fill(4, {3.0.rand});
{ [1, 1] * Klank.ar(`[[200, 400, 500, 350], nil, ~dels], Impulse.ar(0, 0, 0.1)) }.play;
)


// SynthDef from 'factory'
(
(1..7).do{|n|
	var name = \klank_ ++ n;
	SynthDef(name, {|out=0, off_dur=2.0|
		var freqs, decays, sig, timer, test_dur=2.0;
		freqs = NamedControl.kr(\freqs, 440.0!n);
		decays = NamedControl.kr(\decays, 0.5!n);
		timer = EnvGen.kr(Env([0,1, 1], [0.001, off_dur -0.001]), doneAction:2);
		sig = Klank.ar(`[freqs, nil, decays], Impulse.ar(0, 0, 0.2));
		sig = Pan2.ar(sig * timer);
		Out.ar(out, sig);
	}).add;
}
);

~freqs = [[[200, 300, 500]], [[250, 350, 450]], [[200, 450, 510]]];
~decays = [[[0.5, 1.3, 1.0]], [[0.75, 2.1, 0.3]], [[0.7, 0.8, 0.3]]];

(
p = Pbind(
	\freqs, Pseq(~freqs, 2),
	\decays, Pseq(~decays, 2),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq([1], 6),
	\off_dur, Pkey(\decays).collect({|x| x[0].maxItem.asSymbol})
).play
);


x = Synth(\klank_1, [\freqs, [440], \decays, [0.5]])

Linen.kr

[1, 2, 3].maxItem