o = Server.default.options;
o.sampleRate = 192000;
o.device

ServerOptions.devices
o.device = "CA DacMagic 200M 2.0"



// ~modes = File.open("/Users/student/Documents/Myers/azure/JSON/modes.JSON", "r");
~modes = File.open("/Users/jon/Documents/2021/azure/JSON/modes.JSON", "r");
~modes = ~modes.readAllString.parseYAML;
~modes = Array.fill(~modes.size, {arg i; ~modes[i].asFloat});

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

~freqs = ~registrate.value(~modes[0][..4], 3, 100)
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
		cutoff.postln;
		x.set(\vols, new_vals);
		x.set(\cutoff, cutoff);
		x.set(\rq, rq);
		0.5.wait;
	}
}.fork;
));

x.set(\freqs, ~registrate.value(~modes[3][..4], 3, 100))



	[4, 5]*[6, 7]


