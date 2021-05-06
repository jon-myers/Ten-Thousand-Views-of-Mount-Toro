o = Server.default.options;
o.sampleRate = 192000;
o.devices

ServerOptions.devices
o.device = "CA DacMagic 200M 2.0"



// ~modes = File.open("/Users/student/Documents/Myers/azure/JSON/modes.JSON", "r");
~modes = File.open("/Users/jon/Documents/2021/azure/JSON/modes.JSON", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});


~modes = File.open("/Users/jon/Documents/2021/azure/test_modes.JSON", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});

~alt_0 = File.open("/Users/jon/Documents/2021/azure/alt_modes_0.JSON", "r");
~alt_0 = ~alt_0.readAllString.parseYAML;
~alt_0 = Array.fill(~alt_0.size, {arg i; ~alt_0[i].asFloat});

~alt_1 = File.open("/Users/jon/Documents/2021/azure/alt_modes_1.JSON", "r");
~alt_1 = ~alt_1.readAllString.parseYAML;
~alt_1 = Array.fill(~alt_1.size, {arg i; ~alt_1[i].asFloat});





~modes_et_all = File.open("/Users/jon/Documents/2021/azure/JSON/modes_and_variations.JSON", "r");
~modes_et_all = ~modes_et_all.readAllString.parseYAML;
~modes_et_all = Array.fill(~modes_et_all.size, {arg i; ~modes_et_all[i].asFloat});

~modes = ~modes_et_all[0]
~alt_0 = ~modes_et_all[1]
~alt_1 = ~modes_et_all[2]



/*~layer2 = File.open("/Users/student/Documents/Myers/azure/test_2nd_layer.JSON", "r");
~layer2 = ~layer2.readAllString.parseYAML;
~layer2 = Array.fill(~layer2.size, {arg i; ~layer2[i].asFloat});*/

~registrate = {|mode=#[0, 0, 0, 0, 0, 0, 0], octs=4, fund=200|
	var group_index = Array.fill(mode.size, {arg i; (octs*(i+1)/(mode.size+1)).floor});
	var proportions = [];
	mode.do({|item, i|
		(octs - group_index[i]).do({|j|
			proportions = proportions.add((2**(j+group_index[i])) * item)});
	});
	fund * proportions;
};

~startSynth = {|freqs, vols|
	var name = 'drone_' ++ freqs.size;
	Synth(name, [\freqs: freqs, \vols: vols, \lag: 1])
}



// define the synthdefs, so as to send freqs / vols of any size.
(
(1..40).do { |n|
	var name = \drone_ ++ n;
	SynthDef(name, {|lag = 0.5|
		var freqs = Lag.kr(NamedControl.kr(\freqs, 1!n), 0.1);
		var vols = Lag.kr(NamedControl.kr(\vols, 1!n), lag);
		var cutoff = Lag.kr(NamedControl.kr(\cutoff, 500), lag);
		var rq = Lag.kr(NamedControl.kr(\rq, 0.65), lag);
		var sig, mixer;

		sig = LFTri.ar(freqs, 0, vols / vols.size);
		sig = BLowPass.ar(sig, cutoff, rq);
		sig = Pan2.ar(Mix.ar(sig));
		// mixer = Mix.fill(2, {|i|
		Out.ar(0, sig);
	}).add
})

~freqs = ~registrate.value(~modes[0][..4], 3, 150);
~vols = Array.fill(~freqs.size, {0.1});
(x = ~startSynth.value(~freqs, ~vols);
(
z = Array.fill(~freqs.size, {Pbrown(0.0, 1.0, 0.1, inf).asStream});
~cutoffStream = 200 * (2 ** Pbrown(0, 2.5, 0.2, inf).asStream);
~rqStream = Pbrown(0.4, 1.0, 0.01 ).asStream;

{var new_vals, cutoff, rq;
	loop {
		new_vals = Array.fill(~freqs.size, {arg i; z[i].next});
		cutoff = ~cutoffStream.next;
		rq = ~rqStream.next;
		x.set(\vols, new_vals);
		x.set(\cutoff, cutoff);
		x.set(\rq, rq);
		0.5.wait;
	}
}.fork;
));

1!2
~note = PatternProxy(Pseq([300 * ~modes[1][0]], inf))
~note.source = Pseq([300 * ~alt_0[3][3]], inf)
~mel = Pbind(
	\freq, ~note,
	\dur, Prand([0.5, 0.35, 0.7], inf),
	\amp, 0.75
).play


~mel.put(\amp, 0.5)
~mel.play
~mel.stop

x.set(\freqs, ~registrate.value(~modes[0][..4], 3, 150))
x.set(\freqs, ~registrate.value(2*~alt_0[0][..4], 3, 150))
x.set(\freqs, ~registrate.value(2*~alt_1[0][..4], 3, 150))

x.set(\freqs, ~registrate.value(0.5*~modes[1][..4], 3, 150))
x.set(\freqs, ~registrate.value(2*~alt_0[1][..4], 3, 150))
x.set(\freqs, ~registrate.value(2*~alt_1[1][..4], 3, 150))

x.set(\freqs, ~registrate.value(0.5*~modes[2][..4], 3, 150))
x.set(\freqs, ~registrate.value(0.5*~alt_0[2][..4], 3, 150))
x.set(\freqs, ~registrate.value(0.5*~alt_1[2][..4], 3, 150))

x.set(\freqs, ~registrate.value(0.5*~modes[3][..4], 3, 150))
x.set(\freqs, ~registrate.value(1*~alt_0[3][..4], 3, 150))
x.set(\freqs, ~registrate.value(0.5*~alt_1[3][..4], 3, 150))


~layer2[3][0]*2 / ~modes[4][0]
8/7

~layer2[0] / ~modes[0]
~modes[1]

	[4, 5]*[6, 7]


