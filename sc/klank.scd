(
~dir = PathName.new(Document.current.dir);
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


~melody = File.open(~jsonPath ++ "melody.JSON", "r");
~melody = ~melody.readAllString.parseYAML;
~m_freqs = Ar ray.fill(~melody.size, {|i| ~melody[i]["note"].asFloat})
~m_durs = Array.fill(~melody.size, {|i| ~melody[i]["rt_dur"].asFloat})

(
SynthDef.new(\melody, {|out=0, freq=200.0, dur=5|
	var sig, env, filter;
	env = EnvGen.kr(Env.new([0.0, 1.0, 0.75, 0.75, 0.0], [0.1, 0.2, dur-1,0.7]), doneAction: 2);
	sig = LFTri.ar(freq) * 0.05625;
	sig = Pan2.ar(sig, 0);
	sig = env * sig;
	sig = BLowPass.ar(sig, 1000);
	Out.ar(out, sig)
}).add;
);

~klanks[0]["decays"]
~klanks[18]["type"]
~klanks.size.do({|i| if (~klanks[i]["type"] == "rest", {i.postln})})

~klanks[3438].keys()
~klanks[3438]["rt_dur"]
~k_amps


"""And as the world unrolled itself for him, so rolleth it together again for him in rings, as the growth of good through evil, as the growth of purpose out of chance."""



// SynthDef from 'factory'
(
(1..7).do{|n|
	var name = \klank_ ++ n;
	SynthDef(name, {|out=0, off_dur=2.0, pan=0.0, transient_dur=0.001, transient_curve=0|
		var freqs, decays, amps, sig, sig2, timer, test_dur=2.0, input, input_env;
		// pan.poll;
		freqs = NamedControl.kr(\freqs, 440.0!n);
		decays = NamedControl.kr(\decays, 0.5!n);
		amps = NamedControl.kr(\amps, 0.5!n);
		timer = EnvGen.kr(Env([0,1, 1], [transient_dur, off_dur - transient_dur], transient_curve), doneAction:2);
		input_env = EnvGen.kr(Env([0, 0.01, 0], [transient_dur, off_dur - transient_dur]));
		input = Impulse.ar(0, 0, 0.2);
		sig = Klank.ar(`[freqs, amps, decays], input);
		// sig2 = 0.125 * Klank.ar(`[freqs, amps, decays], Impulse.ar(0, 0, 0.2), 2);
		sig = Pan2.ar((sig) * timer, pan);
		Out.ar(out, sig);
	}).add;
}
);

Synth(\klank_3, [\freqs, [200, 300, 500], \decays, [5, 4, 6], \transient_curve, 0, \off_dur,20]);



(
p = Pbind(
	\freqs, Pseq(~k_freqs, 1),
	\decays, Pseq(~k_decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~k_durs, 1),
	\off_dur, Pseq(8*(~k_next_durs + ~k_durs), 1),
	\pan,Pseq(~k_pan, 1),
	\amps, Pseq(~k_amps, 1),
	\transient_dur, Pseq(~k_transient_durs, 1),
	\transient_curve, Pseq(~k_transient_curve, 1)
).play
);

(
m = Pbind(
	\instrument, \melody,
	\freq, Pseq(~m_freqs, 1),
	\dur, Pseq(~m_durs, 1),
).play
)
