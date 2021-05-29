(~dir = PathName.new(Document.current.dir);
~jsonPath = ~dir.fullPath[..~dir.fullPath.size-3] ++ 'JSON/';


~klanks = File.open(~jsonPath ++ "klank_packets.JSON", "r");
~klanks = ~klanks.readAllString.parseYAML;

~freqs = Array.fill(~klanks.size, {|i| [~klanks[i]["freqs"].asFloat]});
~amps = Array.fill(~klanks.size, {|i| [~klanks[i]["amps"].asFloat]});
~pan = Array.fill(~klanks.size, {|i| ~klanks[i]["pan"].asFloat});

~durs = Array.fill(~klanks.size, {|i|
	var dur = ~klanks[i]["rt_dur"];
	dur = if(dur[0].asString=="R", {dur.interpret}, {dur.asFloat});
	dur});

~next_durs = Array.fill(~klanks.size - 1, {|i|
	var dur = ~klanks[i+1]["rt_dur"];
	dur = if(dur[0].asString=="R", {dur.interpret}, {dur.asFloat})});
~next_durs = ~next_durs.add( ~durs[~durs.size - 1]);

~decays = Array.fill(~klanks.size, {|i| [~klanks[i]["rt_decays"]].asFloat});

~transient_durs = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_dur"].asFloat});
~transient_curve = Array.fill(~klanks.size, {|i| ~klanks[i]["transient_curve"].asFloat});
);



// SynthDef from 'factory'
(
(1..7).do{|n|
	var name = \klank_ ++ n;
	SynthDef(name, {|out=0, off_dur=2.0, pan=0.0, transient_dur=0.001, transient_curve=0|
		var freqs, decays, amps, sig, sig2, timer, test_dur=2.0;
		// pan.poll;
		freqs = NamedControl.kr(\freqs, 440.0!n);
		decays = NamedControl.kr(\decays, 0.5!n);
		amps = NamedControl.kr(\amps, 0.5!n);
		timer = EnvGen.kr(Env([0,1, 1], [transient_dur, off_dur - transient_dur], transient_curve), doneAction:2);
		sig = Klank.ar(`[freqs, amps, decays], Impulse.ar(0, 0, 0.2));
		// sig2 = 0.125 * Klank.ar(`[freqs, amps, decays], Impulse.ar(0, 0, 0.2), 2);
		sig = Pan2.ar((sig) * timer, pan);
		Out.ar(out, sig);
	}).add;
}
);
Array
/*~freqs = [[[200, 300, 500]], [[250, 350, 450]], [[200, 450, 510]]];
~decays = [[[0.5, 1.3, 1.0]], [[0.75, 2.1, 0.3]], [[0.7, 0.8, 0.3]]];*/
// ~pan = Array.fill(~freqs.size, {2.0.rand - 1});
(
p = Pbind(
	\freqs, Pseq(~freqs, 1),
	\decays, Pseq(~decays, inf),
	\size, Pkey(\freqs).collect({|x| x[0].size.asSymbol}),
	\instrument, Pkey(\size).collect({|x| \klank_ ++ x}),
	\dur, Pseq(~durs, 1),
	\off_dur, Pseq(8*(~next_durs + ~durs), 1),
	\pan,Pseq(~pan, 1),
	\amps, Pseq(~amps, 1),
	\transient_dur, Prand([0.005], inf),
	\transient_curve, Prand([-4, -3, -2, -1, 0, 1, 2, 3, 4], inf)
).play
);

Rest(10)

~freqs[0][0][0]
~next_durs
~durs
~next_durs + ~durs
a = "Rest(10)"
a
z=a.interpret
z == Rest(10)
Rest(10)